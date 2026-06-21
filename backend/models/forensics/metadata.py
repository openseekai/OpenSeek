
import PIL.ExifTags
import PIL.Image


class MetadataBranch:
    """
    Forensic Branch: Metadata Scoring.
    Authentic photos usually have rich EXIF data (GPS, Model, Settings).
    AI photos usually have stripped or generic metadata.
    """
    REAL_TAGS = [
        "Make", "Model", "ExposureTime", "FNumber",
        "ISOSpeedRatings", "DateTimeOriginal", "LensModel"
    ]

    @staticmethod
    def analyze(image_path: str) -> float:
        """
        Calculate a metadata authenticity score (0-1).
        1.0 = Highly likely real (rich EXIF).
        0.0 = Highly likely AI/Stripped.
        """
        score = 0.0
        try:
            img = PIL.Image.open(image_path)
            exif = img.getexif()
            if not exif:
                return 0.1 # Stripped metadata is common in web images, but suspicious for raw scans

            found_tags = 0
            for tag_id, _value in exif.items():
                tag = PIL.ExifTags.TAGS.get(tag_id, tag_id)
                if tag in MetadataBranch.REAL_TAGS:
                    found_tags += 1

            # Score based on how many "Real Photography" tags are present
            score = min(found_tags / len(MetadataBranch.REAL_TAGS), 1.0)

            # Bonus for GPS data (extremely rare in AI)
            if hasattr(img, 'gps_dict') or "GPSInfo" in exif:
                score = min(score + 0.3, 1.0)

            # Inverse: Penalty for AI software traces
            from utils.forensics import MetadataAnalyzer
            ai_meta = MetadataAnalyzer.scan(image_path)
            if ai_meta["has_ai_metadata"]:
                score = 0.0

        except Exception:
            score = 0.1

        # Return 0 to 1 score (1 = Real, 0 = AI for fusion logic consistency)
        # But wait, user wants metadata_score where High = AI?
        # Requirement: final_score = 0.4*spatial + 0.3*freq + 0.2*noise + 0.1*meta
        # If final_score > threshold: AI-generated.
        # This means all individual scores should be high for AI.
        # So I will invert it.
        return 1.0 - score
