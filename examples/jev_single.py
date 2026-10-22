import base64
import os
from pathlib import Path
from uuid import uuid4

import requests
from dotenv import load_dotenv
from PIL import Image


# ==================== 設定 ====================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    PROJECT_ROOT
    / "assets"
    / "frame_13.png"
)

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.14.0"

# 存在機率的示範門檻，請依你的資料調整。
MIN_PRESENCE = 0.7

# 英文 ID 用於程式取值；中文名稱用於結果顯示。
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
        "description": "具有挖掘臂與鏟斗的挖掘工程機械。",
    },
    "worker": {
        "label": "工地人員",
        "description": "施工現場的工作人員；定位時框住整個可見的人。",
    },
    "loader": {
        "label": "鏟裝機",
        "description": "前方具有大型裝載鏟斗的工程車輛，不是挖土機。",
    },
    "traffic_cone": {
        "label": "交通錐",
        "description": "道路或工地用於警示、引導交通的錐形物。",
    },
    "water_barrier": {
        "label": "注水式護欄",
        "description": "可注水的塑膠交通隔離護欄，常見紅白色。",
    },
    "fence": {
        "label": "圍欄",
        "description": "用於圍起或隔離施工區域的圍籬或欄杆。",
    },
}


# ==================== 圖片處理 ====================

def prepare_image(image_path: Path) -> tuple[str, str, int, int]:
    """確認圖片規格，回傳格式、Base64、寬、高。"""
    if not image_path.is_file():
        raise FileNotFoundError(f"找不到圖片：{image_path}")

    with Image.open(image_path) as image:
        width, height = image.size
        image_format = image.format

        if image_format not in {"JPEG", "PNG"}:
            raise ValueError("圖片必須是 JPEG 或 PNG。")

        if getattr(image, "is_animated", False):
            raise ValueError("不支援動畫圖片，請改用靜態 PNG 或 JPEG。")

        if not (280 <= width <= 1568 and 280 <= height <= 1568):
            raise ValueError(
                f"圖片尺寸為 {width} × {height}；"
                "寬、高都必須介於 280～1568 像素。"
            )

        image.load()

    image_bytes = image_path.read_bytes()

    if len(image_bytes) > 16 * 1024 * 1024:
        raise ValueError("單張圖片不可超過 16 MiB。")

    fmt = "jpg" if image_format == "JPEG" else "png"
    encoded = base64.b64encode(image_bytes).decode("ascii")

    return fmt, encoded, width, height


# ==================== 建立問題 ====================

def build_questions() -> dict:
    questions = {
        "photo_category": {
            "type": "choice",
            "instructions": (
                "判斷 `photo` 中最主要、最突出的主體。"
                "若有多種物件，只選最能代表圖片主體的一類；"
                "若都不符合，選擇 other。"
            ),
            "criteria": {
                category_id: (
                    f"{item['label']}：{item['description']}"
                )
                for category_id, item in CATEGORIES.items()
            },
        },
    }

    questions["photo_category"]["criteria"]["other"] = (
        "其他或無法判斷：圖片主體不屬於上述類別，或無法清楚辨識。"
    )

    for category_id, item in CATEGORIES.items():
        label = item["label"]
        description = item["description"]

        questions[f"{category_id}_visible"] = {
            "type": "noul",
            "instructions": (
                f"`photo` 中是否至少有一個可辨識的{label}？"
                f"定義：{description}"
            ),
        }

        questions[f"{category_id}_box"] = {
            "type": "box",
            "instructions": (
                f"找出 `photo` 中最明顯的一個{label}。"
                f"定義：{description}"
                "若有多個，只定位最明顯的一個。"
                "框住該物件的完整可見範圍，不要框住整張圖片。"
            ),
        }

    return questions


# ==================== 呼叫 TypeSafe ====================

def analyze_image(api_key: str, fmt: str, encoded: str) -> dict:
    image_ref = str(uuid4())

    payload = {
        "model": MODEL,
        "state": {
            "photo": image_ref,
        },
        "assets": {
            image_ref: {
                "type": "image",
                "format": fmt,
                "data_base64": encoded,
            },
        },
        "questions": build_questions(),
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "TypeSafe-Danger-Zone-Images": "true",
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json=payload,
        timeout=(15, 180),
    )

    if not response.ok:
        raise RuntimeError(
            f"TypeSafe API 回傳 HTTP {response.status_code}\n"
            f"{response.text}"
        )

    result = response.json()

    if not isinstance(result.get("answers"), dict):
        raise RuntimeError(f"API 回傳缺少 answers：{result}")

    return result


# ==================== 顯示結果 ====================

def print_results(result: dict, width: int, height: int) -> None:
    answers = result["answers"]

    category_answer = answers["photo_category"]
    category_id = category_answer["choice"]

    if category_id in CATEGORIES:
        main_label = CATEGORIES[category_id]["label"]
    else:
        main_label = "其他或無法判斷"

    print(f"主要分類：{main_label}")
    print(f"分類信心值：{category_answer['confidence']:.1%}")
    print(f"圖片尺寸：{width} × {height}")
    print()
    print("各類別定位結果：")

    found_count = 0

    for category_id, item in CATEGORIES.items():
        presence = answers[f"{category_id}_visible"]["noul"]

        # Box 即使目標不存在也會回傳框，因此先檢查存在機率。
        if presence < MIN_PRESENCE:
            continue

        box_answer = answers[f"{category_id}_box"]
        edges = box_answer["box"]

        left = round(edges["left"] * width)
        top = round(edges["top"] * height)
        right = round(edges["right"] * width)
        bottom = round(edges["bottom"] * height)

        print(
            f"- {item['label']}："
            f"存在機率 {presence:.1%}，"
            f"定位信心值 {box_answer['confidence']:.1%}"
        )
        print(
            f"  像素座標："
            f"left={left}, top={top}, "
            f"right={right}, bottom={bottom}"
        )
        print(
            f"  比例座標："
            f"left={edges['left']:.3f}, "
            f"top={edges['top']:.3f}, "
            f"right={edges['right']:.3f}, "
            f"bottom={edges['bottom']:.3f}"
        )

        found_count += 1

    if found_count == 0:
        print("沒有類別的存在機率達到設定門檻。")


# ==================== 程式入口 ====================

def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("TYPESAFE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "請在專案根目錄的 .env 設定 TYPESAFE_API_KEY。"
        )

    fmt, encoded, width, height = prepare_image(IMAGE_PATH)
    result = analyze_image(api_key, fmt, encoded)
    print_results(result, width, height)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.Timeout:
        raise SystemExit("執行失敗：API 請求逾時，請稍後重試。") from None
    except requests.exceptions.ConnectionError:
        raise SystemExit(
            "執行失敗：無法連線至 TypeSafe，請檢查網路與 DNS。"
        ) from None
    except Exception as exc:
        raise SystemExit(f"執行失敗：{exc}") from None