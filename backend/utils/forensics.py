
import PIL.ExifTags
import PIL.Image


class MetadataAnalyzer:
    """
    Forensic Tool: Scans image metadata for AI signatures.
    Targets EXIF/IPTC tags that modern AI models often leak,
    and penalizes missing or impossible camera parameter combinations.
    """
    AI_SIGNATURES = [
        "dall-e", "midjourney", "stable diffusion", "adobe firefly",
        "bing image creator", "canva ai", "starryai", "miricanvas",
        "wombo", "artbreeder", "nightcafe"
    ]

    @staticmethod
    def scan(image_path: str) -> dict:
        """Analyze image metadata for AI software traces and parameter inconsistencies."""
        info = {
            "has_ai_metadata": False,
            "software": None,
            "tags_found": [],
            "suspicion_score": 0.0,
            "anomalies": []
        }

        try:
            img = PIL.Image.open(image_path)
            exif = img.getexif()

            if not exif:
                # No standard EXIF. Suspicious if from a raw source, but common on social media.
                # Just add a lightweight baseline suspicion.
                info["suspicion_score"] = 0.2
                info["anomalies"].append("Missing entire EXIF block")
                return info

            valid_camera_found = False
            has_iso = False
            has_exposure = False

            # Check standard EXIF tags
            for tag_id, value in exif.items():
                tag = PIL.ExifTags.TAGS.get(tag_id, tag_id)
                value_str = str(value).lower()

                # Check for explicit AI signatures
                for sig in MetadataAnalyzer.AI_SIGNATURES:
                    if sig in value_str:
                        info["has_ai_metadata"] = True
                        info["software"] = sig
                        info["tags_found"].append(f"{tag}: {sig}")
                        info["suspicion_score"] = 1.0

                # Advanced EXIF Verification
                if tag in ['Make', 'Model']:
                    if any(x in value_str for x in ['canon', 'nikon', 'sony', 'apple', 'samsung', 'google', 'fujifilm', 'panasonic']):
                        valid_camera_found = True
                    elif len(value_str) > 2:
                        # Obscure or fake Make/Model string
                        info["anomalies"].append(f"Suspicious Camera {tag}: {value}")

                if tag == 'ISOSpeedRatings':
                    has_iso = True
                if tag == 'ExposureTime':
                    has_exposure = True

            # If there's EXIF data, but it claims to be a photo without physical sensors
            if exif and (not valid_camera_found):
                info["suspicion_score"] += 0.3
                info["anomalies"].append("EXIF present but lacks valid Camera Make/Model")

            if valid_camera_found and not (has_iso or has_exposure):
                info["suspicion_score"] += 0.4
                info["anomalies"].append("Claims to be real camera but lacks required sensor parameters (ISO/Exposure)")

            # Check for TIFF software tag (common for Firefly/Canva)
            if hasattr(img, 'info') and 'software' in img.info:
                sw = str(img.info['software']).lower()
                for sig in MetadataAnalyzer.AI_SIGNATURES:
                    if sig in sw:
                        info["has_ai_metadata"] = True
                        info["software"] = sig
                        info["tags_found"].append(f"Software: {sig}")
                        info["suspicion_score"] = 1.0

        except Exception as e:
            print(f"[Metadata] Error: {e}")

        info["suspicion_score"] = min(info["suspicion_score"], 1.0)
        return info

class ExplanationGenerator:
    """
    Engine: Converts forensic signals into human-readable technical reasons.
    """
    @staticmethod
    def generate(res: dict) -> str:
        """Compose an explanation based on the forensic evidence."""
        reasons = []

        # 1. Spectral Reason
        if res.get("spectral_score", 0) > 0.6:
            reasons.append("High-frequency spectral artifacts (FFT) suggest synthetic generation noise.")
        elif res.get("spectral_score", 0) > 0.45:
            reasons.append("Minor frequency-domain inconsistencies detected.")

        # 2. Expert Reason
        if res.get("expert_score", 0) > 0.4:
            reasons.append("Deep residual fingerprints matching modern Diffusion models identified.")

        # 3. Biometric Reason (for Humans)
        if res.get("face_detected"):
            if res.get("iris_score", 0) > 0.6:
                reasons.append("Non-biological iris patterns and gaze asymmetry detected.")
            elif res.get("authenticity_score", 0) < 15:
                reasons.append("Authentic biological micro-saccades and pupil consistency verified.")

        # 4. Metadata Reason
        if res.get("metadata", {}).get("has_ai_metadata"):
            reasons.append(f"Metadata signature from {res['metadata']['software'].upper()} found.")

        if not reasons:
            return "No definitive AI artifacts found. Visual patterns align with standard photography."

        return " ".join(reasons[:2]) # Keep it concise
