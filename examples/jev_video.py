import base64
import math
import os
import queue
import threading
import time
from pathlib import Path
from uuid import uuid4

import cv2
import requests
from dotenv import load_dotenv


# ==================== 設定 ====================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = PROJECT_ROOT / "assets" / "black_cat_walking.mp4"

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.14.0"

WINDOW_NAME = "Video + Jev"

# 影片時間每隔幾秒，嘗試送出一張影格。
# 若上一個 API 請求尚未完成，則略過，不累積請求。
SAMPLE_INTERVAL_SECONDS = 2.0

# 物件存在機率達到門檻，才顯示框。
MIN_PRESENCE = 0.7

# 框所屬影格與目前影片畫面相差超過幾秒，就隱藏框。
MAX_BOX_AGE_SECONDS = 3.0

# 圖片等比例縮放後，補邊成此尺寸。
IMAGE_SIZE = 960

# API 連線／讀取逾時設定。
API_TIMEOUT = (15, 120)

CATEGORIES = {
    "vest": {
        "label": "反光背心",
        "description": "穿戴於身上的高可見度背心，通常有反光條。",
    },
    "helmet": {
        "label": "安全帽",
        "description": "工地使用的頭部防護安全帽。",
    },
    "excavator": {
        "label": "挖土機",
        "description": "具有長挖掘臂與鏟斗的挖掘工程機械。",
    },
    "worker": {
        "label": "工地人員",
        "description": (
            "施工現場的工作人員；"
            "不要只因為有人出現，就假定為工地人員。"
        ),
    },
    "loader": {
        "label": "鏟裝機",
        "description": "具有大型前置裝載鏟斗的工程車輛，不是挖土機。",
    },
    "traffic_cone": {
        "label": "交通錐",
        "description": "道路或工地使用的錐形警示物。",
    },
    "water_barrier": {
        "label": "注水式護欄",
        "description": "可注水的塑膠交通隔離護欄，常見紅白色。",
    },
    "fence": {
        "label": "圍欄",
        "description": "隔離區域使用的圍籬或欄杆。",
    },
}

# OpenCV 的顏色順序為 BGR。
COLORS = {
    "vest": (0, 255, 255),
    "helmet": (0, 165, 255),
    "excavator": (255, 180, 0),
    "worker": (0, 255, 0),
    "loader": (255, 0, 255),
    "traffic_cone": (0, 100, 255),
    "water_barrier": (255, 255, 0),
    "fence": (180, 100, 255),
}


# ==================== 建立問題 ====================

def build_questions() -> dict:
    questions = {}

    for category_id, item in CATEGORIES.items():
        label = item["label"]
        description = item["description"]

        questions[f"{category_id}_visible"] = {
            "type": "noul",
            "instructions": (
                f"`photo` 中是否至少有一個可辨識的{label}？"
                f"定義：{description}"
                "只依據可見內容，不要根據場景猜測物件必然存在。"
            ),
        }

        questions[f"{category_id}_box"] = {
            "type": "box",
            "instructions": (
                f"框住 `photo` 中最明顯的一個{label}。"
                f"定義：{description}"
                "若有多個，選可見面積最大的目標；"
                "若大小相近，選最接近畫面中央者。"
                "框住該物件完整的可見範圍。"
            ),
        }

    return questions


# ==================== 影格處理 ====================

def prepare_frame(frame):
    height, width = frame.shape[:2]

    scale = IMAGE_SIZE / max(width, height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))

    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=(
            cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        ),
    )

    pad_left = (IMAGE_SIZE - resized_width) // 2
    pad_top = (IMAGE_SIZE - resized_height) // 2

    prepared = cv2.copyMakeBorder(
        resized,
        pad_top,
        IMAGE_SIZE - resized_height - pad_top,
        pad_left,
        IMAGE_SIZE - resized_width - pad_left,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    success, jpeg = cv2.imencode(
        ".jpg",
        prepared,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )

    if not success:
        raise RuntimeError("影格 JPEG 編碼失敗。")

    image_bytes = jpeg.tobytes()

    if len(image_bytes) > 16 * 1024 * 1024:
        raise ValueError("圖片超過 16 MiB。")

    encoded = base64.b64encode(image_bytes).decode("ascii")

    geometry = {
        "width": width,
        "height": height,
        "scale_x": resized_width / width,
        "scale_y": resized_height / height,
        "pad_left": pad_left,
        "pad_top": pad_top,
    }

    return encoded, geometry


def restore_box(edges: dict, geometry: dict):
    values = {
        name: float(edges[name])
        for name in ("left", "top", "right", "bottom")
    }

    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Box 包含無效座標。")

    if not (
        0 <= values["left"] < values["right"] <= 1
        and 0 <= values["top"] < values["bottom"] <= 1
    ):
        raise ValueError("Box 座標範圍或順序不正確。")

    def convert(value, padding, scale, limit):
        pixel = (value * IMAGE_SIZE - padding) / scale
        return max(0, min(limit, round(pixel)))

    left = convert(
        values["left"],
        geometry["pad_left"],
        geometry["scale_x"],
        geometry["width"],
    )
    right = convert(
        values["right"],
        geometry["pad_left"],
        geometry["scale_x"],
        geometry["width"],
    )
    top = convert(
        values["top"],
        geometry["pad_top"],
        geometry["scale_y"],
        geometry["height"],
    )
    bottom = convert(
        values["bottom"],
        geometry["pad_top"],
        geometry["scale_y"],
        geometry["height"],
    )

    if left >= right or top >= bottom:
        return None

    return left, top, right, bottom


# ==================== 呼叫 TypeSafe ====================

def analyze_image(session, encoded: str) -> dict:
    image_ref = str(uuid4())

    payload = {
        "model": MODEL,
        "state": {
            "photo": image_ref,
        },
        "assets": {
            image_ref: {
                "type": "image",
                "format": "jpg",
                "data_base64": encoded,
            },
        },
        "questions": build_questions(),
    }

    response = session.post(
        API_URL,
        json=payload,
        timeout=API_TIMEOUT,
    )

    if not response.ok:
        raise RuntimeError(
            f"TypeSafe API HTTP {response.status_code}\n"
            f"{response.text[:2000]}"
        )

    result = response.json()

    if not isinstance(result.get("answers"), dict):
        raise RuntimeError(f"API 回傳缺少 answers：{result}")

    return result


def parse_detections(result: dict, geometry: dict) -> list:
    answers = result["answers"]
    detections = []

    for category_id, item in CATEGORIES.items():
        presence = float(
            answers[f"{category_id}_visible"]["noul"]
        )

        if not math.isfinite(presence) or not 0 <= presence <= 1:
            raise ValueError(
                f"{category_id} 的存在機率不是有效的 0～1 數值。"
            )

        if presence < MIN_PRESENCE:
            continue

        box_answer = answers[f"{category_id}_box"]
        confidence = float(box_answer["confidence"])

        if (
            not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError(
                f"{category_id} 的定位信心值無效。"
            )

        bbox = restore_box(box_answer["box"], geometry)

        if bbox is None:
            continue

        detections.append({
            "class": category_id,
            "label": item["label"],
            "bbox": bbox,
            "presence": presence,
            "box_confidence": confidence,
        })

    return detections


# ==================== 背景分析 ====================

def analysis_worker(api_key, jobs, outputs, stop_event):
    with requests.Session() as session:
        session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "TypeSafe-Danger-Zone-Images": "true",
        })

        while not stop_event.is_set():
            try:
                frame, timestamp = jobs.get(timeout=0.1)
            except queue.Empty:
                continue

            if stop_event.is_set():
                break

            try:
                started = time.monotonic()

                encoded, geometry = prepare_frame(frame)
                result = analyze_image(session, encoded)
                detections = parse_detections(result, geometry)

                message = {
                    "timestamp": timestamp,
                    "detections": detections,
                    "elapsed": time.monotonic() - started,
                    "error": None,
                }

            except Exception as exc:
                message = {
                    "timestamp": timestamp,
                    "error": str(exc),
                }

            if not stop_event.is_set():
                outputs.put(message)


# ==================== 畫框與顯示 ====================

def put_text(image, text, position, color=(255, 255, 255)):
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_detections(frame, detections: list):
    output = frame.copy()
    height, width = output.shape[:2]

    for det in detections:
        left, top, right, bottom = det["bbox"]

        left = max(0, min(width - 1, left))
        right = max(0, min(width - 1, right))
        top = max(0, min(height - 1, top))
        bottom = max(0, min(height - 1, bottom))

        if left >= right or top >= bottom:
            continue

        color = COLORS[det["class"]]

        cv2.rectangle(
            output,
            (left, top),
            (right, bottom),
            color,
            2,
        )

        # 預設字型使用英文；中文結果印在命令列。
        label = f"{det['class']} p={det['presence']:.2f}"

        put_text(
            output,
            label,
            (left, max(20, top - 6)),
            color,
        )

    return output


def print_results(message: dict):
    print(
        f"\n[影片影格 {message['timestamp']:.2f} 秒] "
        f"分析耗時 {message['elapsed']:.2f} 秒",
        flush=True,
    )

    if not message["detections"]:
        print("  沒有可顯示的候選框；不代表確定沒有物件。")
        return

    for det in message["detections"]:
        print(
            f"  - {det['label']}："
            f"存在機率 {det['presence']:.1%}，"
            f"定位信心值 {det['box_confidence']:.1%}，"
            f"原圖座標 {det['bbox']}"
        )


# ==================== 影片播放 ====================

def process_video(api_key: str):
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"找不到影片：{VIDEO_PATH}")

    capture = cv2.VideoCapture(str(VIDEO_PATH))
    stop_event = threading.Event()

    jobs = queue.Queue(maxsize=1)
    outputs = queue.Queue()

    try:
        if not capture.isOpened():
            raise RuntimeError(f"無法開啟影片：{VIDEO_PATH}")

        fps = capture.get(cv2.CAP_PROP_FPS)

        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError("無法取得有效影片 FPS。")

        worker = threading.Thread(
            target=analysis_worker,
            args=(api_key, jobs, outputs, stop_event),
            daemon=True,
        )
        worker.start()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1100, 700)

        frame_index = 0
        next_sample_time = 0.0
        pending = False
        latest = None
        started = time.monotonic()

        print(f"影片：{VIDEO_PATH}")
        print(f"FPS：{fps:.3f}")
        print("開始播放，按 Q 或 Esc 結束。", flush=True)

        while True:
            success, frame = capture.read()

            if not success:
                if frame_index == 0:
                    raise RuntimeError("無法讀取第一張影格。")

                print("\n影片結束，或解碼器無法繼續讀取。")
                break

            timestamp = frame_index / fps
            frame_index += 1

            # 固定幀率影片的近似播放排程。
            target_time = started + timestamp

            while time.monotonic() < target_time:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return

            # 接收背景分析結果。
            while True:
                try:
                    message = outputs.get_nowait()
                except queue.Empty:
                    break

                pending = False

                if message["error"]:
                    raise RuntimeError(message["error"])

                latest = message
                print_results(message)

            # 同時最多一個請求，不累積待分析影格。
            if timestamp >= next_sample_time and not pending:
                jobs.put_nowait((frame.copy(), timestamp))
                pending = True

                next_sample_time = (
                    timestamp + SAMPLE_INTERVAL_SECONDS
                )

            # 在目前畫面上，疊加尚未過期的歷史框。
            display = frame.copy()

            if latest is None:
                status = "Waiting for first Jev result..."
            else:
                age = max(
                    0.0,
                    timestamp - latest["timestamp"],
                )

                if age <= MAX_BOX_AGE_SECONDS:
                    display = draw_detections(
                        frame,
                        latest["detections"],
                    )

                    status = (
                        f"Jev source: {latest['timestamp']:.2f}s"
                        f" | age: {age:.2f}s"
                    )

                    if not latest["detections"]:
                        status += " | no boxes"
                else:
                    status = (
                        f"Result age: {age:.2f}s"
                        " | expired - boxes hidden"
                    )

            if pending:
                status += " | analyzing..."

            # 狀態資訊放在獨立頂端欄，不遮住原始畫面。
            display = cv2.copyMakeBorder(
                display,
                70,
                0,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

            put_text(
                display,
                f"Video: {timestamp:.2f}s | Q / Esc: quit",
                (12, 25),
            )

            put_text(
                display,
                status,
                (12, 52),
                (0, 255, 255),
            )

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            if cv2.getWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_VISIBLE,
            ) < 1:
                break

        print("播放完成，關閉視窗。", flush=True)

    finally:
        stop_event.set()
        capture.release()
        cv2.destroyAllWindows()


# ==================== 程式入口 ====================

def main():
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("TYPESAFE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "請在專案根目錄的 .env 設定 TYPESAFE_API_KEY。"
        )

    if (
        not math.isfinite(SAMPLE_INTERVAL_SECONDS)
        or SAMPLE_INTERVAL_SECONDS <= 0
    ):
        raise ValueError("SAMPLE_INTERVAL_SECONDS 必須大於 0。")

    if (
        not math.isfinite(MAX_BOX_AGE_SECONDS)
        or MAX_BOX_AGE_SECONDS <= 0
    ):
        raise ValueError("MAX_BOX_AGE_SECONDS 必須大於 0。")

    if not 0 <= MIN_PRESENCE <= 1:
        raise ValueError("MIN_PRESENCE 必須介於 0～1。")

    if not 280 <= IMAGE_SIZE <= 1568:
        raise ValueError("IMAGE_SIZE 必須介於 280～1568。")

    process_video(api_key)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已停止。")
    except Exception as exc:
        raise SystemExit(f"執行失敗：{exc}") from None