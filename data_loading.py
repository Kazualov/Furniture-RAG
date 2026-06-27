import pandas as pd
from datasets import load_dataset


def extract_image_url(images_field):
    """
    Safely extracts the best available image URL.
    Preference:
    1. hi_res
    2. large
    """
    if not images_field:
        return None

    if isinstance(images_field, dict):
        hi_res_list = images_field.get("hi_res")

        if isinstance(hi_res_list, list):
            for url in hi_res_list:
                if url:
                    return url

        large_list = images_field.get("large")

        if isinstance(large_list, list):
            for url in large_list:
                if url:
                    return url

    elif isinstance(images_field, list):
        for url in images_field:
            if isinstance(url, str) and url:
                return url

    return None


def safe_join(field):
    """
    Converts list/string fields into clean text.
    """
    if isinstance(field, list):
        return " ".join(
            str(x).strip()
            for x in field
            if x and str(x).strip()
        )

    if field:
        return str(field).strip()

    return ""


def details_to_text(details):
    """
    Converts Amazon details dict into text.

    Example:
    {
        "Brand": "HON",
        "Color": "Black"
    }

    ->
    "Brand: HON Color: Black"
    """
    if not isinstance(details, dict):
        return ""

    parts = []

    for key, value in details.items():
        if value:
            parts.append(f"{key}: {value}")

    return " ".join(parts)


def build_parquet_fast():
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_meta_Office_Products",
        split="full",
        trust_remote_code=True
    )

    print(
        f"Successfully downloaded {len(dataset)} raw products. "
        f"Starting cleaning..."
    )

    clean_data = []

    for item in dataset:

        parent_asin = item.get("parent_asin")

        title = str(
            item.get("title", "")
        ).strip()

        description = safe_join(
            item.get("description")
        )

        features = safe_join(
            item.get("features")
        )

        categories_raw = item.get(
            "categories", []
        )

        categories = " > ".join(
            str(x).strip()
            for x in categories_raw
            if x
        )

        details_text = details_to_text(
            item.get("details")
        )

        image_url = extract_image_url(
            item.get("images")
        )

        price = item.get("price")

        average_rating = item.get(
            "average_rating"
        )

        rating_number = item.get(
            "rating_number"
        )

        store = str(
            item.get("store", "")
        ).strip()

        # Basic filtering

        if not parent_asin:
            continue

        if not title:
            continue

        # Rich text for embeddings

        full_text = f"""
Title: {title}

Categories: {categories}

Features:
{features}

Details:
{details_text}

Description:
{description}
""".strip()

        if len(full_text) < 20:
            continue

        clean_data.append({
            "parent_asin": parent_asin,

            "title": title,

            "description": description,

            "features": features,

            "categories": categories,

            "details_text": details_text,

            "price": price,

            "average_rating": average_rating,

            "rating_number": rating_number,

            "store": store,

            "image_url": image_url,

            "full_text": full_text
        })

    print(
        f"Cleaning completed! "
        f"Obtained {len(clean_data)} valid products."
    )

    df = pd.DataFrame(clean_data)

    df.to_parquet(
        "office_products_full.parquet",
        engine="pyarrow",
        index=False
    )

    print(
        "Saved full dataset: "
        "office_products_full.parquet"
    )

    df.head(5000).to_parquet(
        "office_products_micro.parquet",
        engine="pyarrow",
        index=False
    )

    print(
        "Saved micro dataset: "
        "office_products_micro.parquet"
    )


if __name__ == "__main__":
    build_parquet_fast()
