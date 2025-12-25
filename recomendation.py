import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==================================================
# 1. Загрузка данных
# ==================================================

reviews = pd.read_csv("reviews.csv")
salons = pd.read_csv("salons.csv")

# ==================================================
# 2. Очистка числовых полей
# ==================================================

# рейтинг
salons["rating"] = (
    salons["rating"]
    .astype(str)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

# количество отзывов ("1093 оценки" -> 1093)
salons["total_reviews_count"] = (
    salons["total_reviews_count"]
    .astype(str)
    .str.extract(r"(\d+)")
    .astype(float)
    .fillna(0)
)

# ==================================================
# 3. Нормализация районов (ключевая часть)
# ==================================================

def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("район", "")
    text = re.sub(r"[^\w\s]", "", text)
    text = text.strip()

    # убираем типичные русские окончания
    for suffix in ["ский", "ском", "ского", "ская", "ской", "ском"]:
        if text.endswith(suffix):
            text = text.replace(suffix, "")
    return text


salons["district_normalized"] = salons["district"].apply(normalize_text)

# карта: нормализованный → оригинальный
district_map = dict(
    zip(salons["district_normalized"], salons["district"])
)

# ==================================================
# 4. Обработка отзывов (негатив штрафуется)
# ==================================================

def build_reviews(group):
    texts = []
    negative = 0

    for _, row in group.iterrows():
        if row["rating"] <= 2:
            texts.append(f"Негативный отзыв: {row['text']}")
            negative += 1
        else:
            texts.append(f"Позитивный отзыв: {row['text']}")

    return " ".join(texts), negative


reviews_agg = (
    reviews
    .groupby("salon_id")
    .apply(
        lambda g: pd.Series(
            build_reviews(g),
            index=["reviews_text", "negative_reviews_count"]
        )
    )
    .reset_index()
)

# ==================================================
# 5. Merge
# ==================================================

data = salons.merge(reviews_agg, on="salon_id", how="left")
data["reviews_text"] = data["reviews_text"].fillna("")
data["negative_reviews_count"] = data["negative_reviews_count"].fillna(0)

# ==================================================
# 6. Текст для embedding
# ==================================================

data["full_text"] = (
    "Салон: " + data["name"] + ". "
    "Район: " + data["district"] + ". "
    "Адрес: " + data["address"] + ". "
    "Рейтинг: " + data["rating"].astype(str) + ". "
    "Отзывы: " + data["reviews_text"]
)

# ==================================================
# 7. Embedding модель
# ==================================================

model = SentenceTransformer("all-MiniLM-L6-v2")

salon_embeddings = model.encode(
    data["full_text"].tolist(),
    show_progress_bar=True
)

# ==================================================
# 8. Определение района из запроса
# ==================================================

def extract_district(query: str):
    q_norm = normalize_text(query)

    for d_norm, original in district_map.items():
        if d_norm and d_norm in q_norm:
            return original

    return None

# ==================================================
# 9. Рекомендательная функция
# ==================================================

def recommend_salon(
    query: str,
    top_k: int = 3,
    min_rating: float = 3.0,
    penalty_weight: float = 0.1
):
    # 🔍 район из запроса
    detected_district = extract_district(query)

    filtered_data = data.copy()
    filtered_embeddings = salon_embeddings

    if detected_district:
        print(f"📍 Район из запроса: {detected_district}")

        mask = (
            filtered_data["district"] == detected_district
        )
        filtered_data = filtered_data[mask]
        filtered_embeddings = salon_embeddings[mask.values]

        if filtered_data.empty:
            return "❌ Нет салонов в этом районе"

    # ⭐ фильтр по рейтингу
    mask_rating = filtered_data["rating"] >= min_rating
    filtered_data = filtered_data[mask_rating]
    filtered_embeddings = filtered_embeddings[mask_rating.values]

    if filtered_data.empty:
        return "❌ Нет салонов с подходящим рейтингом"

    # 🧠 embedding запроса
    query_emb = model.encode([query])

    similarities = cosine_similarity(
        query_emb,
        filtered_embeddings
    )[0]

    filtered_data = filtered_data.copy()
    filtered_data["base_score"] = similarities

    # 🔻 штраф за негатив
    filtered_data["penalty"] = np.where(
        filtered_data["total_reviews_count"] > 0,
        (filtered_data["negative_reviews_count"] /
         filtered_data["total_reviews_count"]) * penalty_weight,
        0
    )

    filtered_data["final_score"] = (
        filtered_data["base_score"] - filtered_data["penalty"]
    )

    return (
        filtered_data
        .sort_values("final_score", ascending=False)
        .head(top_k)
        [["name", "district", "rating", "final_score"]]
    )

# ==================================================
# 10. Пример запуска
# ==================================================

if __name__ == "__main__":
    queries = [
        "выдай мне салон в бостандыкский район",
        "ищу салон в бостандыкском районе для ресниц",
        "аккуратное наращивание ресниц в алмалинском",
        "хочу хороший салон"
    ]

    for q in queries:
        print("\n🔍", q)
        print(recommend_salon(q))
