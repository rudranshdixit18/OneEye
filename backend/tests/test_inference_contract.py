from backend.app.services.inference import MINIMUM_ALERT_CONFIDENCE, InferenceEngine


def test_alert_thresholds_are_never_below_ninety_percent(tmp_path):
    engine = InferenceEngine(tmp_path, {"fight": 0.2})
    assert engine.thresholds["fight"] == MINIMUM_ALERT_CONFIDENCE
    assert MINIMUM_ALERT_CONFIDENCE > 0.90
    engine.set_thresholds({"fight": 0.5})
    assert engine.thresholds["fight"] == MINIMUM_ALERT_CONFIDENCE


def test_shared_checkpoint_class_filter(tmp_path):
    engine = InferenceEngine(tmp_path, {})
    names = {0: "Accident", 1: "Fall-Detected", 2: "Normal"}
    assert engine._label_matches("accident", "Accident", names)
    assert not engine._label_matches("accident", "Fall-Detected", names)
    assert engine._label_matches("fallen_person", "Fall-Detected", names)


def test_negative_class_rejected(tmp_path):
    engine = InferenceEngine(tmp_path, {})
    assert not engine._positive_class("fight", "NonFight", {0: "Fight", 1: "NonFight"})
    assert not engine._positive_class("fight", "un_fight", {0: "fight", 1: "un_fight"})
    assert engine._positive_class("fight", "Fight", {0: "Fight", 1: "NonFight"})


def test_temporal_models_are_discovered_from_loaded_status(tmp_path):
    engine = InferenceEngine(tmp_path, {})
    engine._status = {
        "fight": {"loaded": True, "temporal_tiles": 4},
        "garbage": {"loaded": True, "temporal_tiles": 1},
        "accident": {"loaded": False, "temporal_tiles": 4},
    }
    assert engine.temporal_event_types() == {"fight"}
