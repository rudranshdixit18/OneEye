from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_frontend_keeps_required_sections_and_models():
    html = (ROOT / "FrontEnd" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "FrontEnd" / "script.js").read_text(encoding="utf-8")
    for section in ("dashboard", "live", "detections", "analytics", "settings"):
        assert f'id="{section}"' in html
    for event_type in ("fight", "garbage", "fallen_person", "mobile_phone", "accident"):
        assert event_type in html or event_type in script
    assert "/api/v1" in script
