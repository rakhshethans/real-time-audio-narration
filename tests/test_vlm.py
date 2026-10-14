# SPDX-License-Identifier: AGPL-3.0-only
import pytest

from audio_narration.vision.vlm import parse_observation


def test_parse_structured_observation():
    item = parse_observation(
        '```json\n{"description":"A person takes a drink.",'
        '"actions":["Drinking"],"uncertain":true}\n```',
        10,
    )
    assert item.actions == ("Drinking",)
    assert item.uncertain


@pytest.mark.parametrize(
    "payload",
    [
        '"A scene"',
        "{}",
        '{"description":"A scene", "facts":"wrong"}',
        '{"description":"A scene", "uncertain":"false"}',
        "not json",
    ],
)
def test_malformed_responses_rejected(payload):
    with pytest.raises(ValueError):
        parse_observation(payload, 0)


def test_identical_clip_cannot_establish_motion():
    import numpy as np

    from audio_narration.domain import Frame, Observation
    from audio_narration.vision.vlm import enforce_temporal_evidence, has_visual_motion

    image = np.zeros((32, 32, 3))
    frames = (Frame(1, 0, image), Frame(2, 1, image.copy()))
    assert not has_visual_motion(frames)
    raw = Observation(
        1,
        "A person is standing. The person walks away.",
        ("A person is standing.",),
        ("The person walks away.",),
        ("The person left.",),
        False,
    )
    checked = enforce_temporal_evidence(raw, False)
    assert checked.description == "A person is standing."
    assert checked.actions == () and checked.changes == () and checked.uncertain


def test_motion_gate_does_not_claim_which_action_happened():
    import numpy as np

    from audio_narration.domain import Frame
    from audio_narration.vision.vlm import has_visual_motion

    assert has_visual_motion(
        (Frame(1, 0, np.zeros((32, 32, 3))), Frame(2, 1, np.ones((32, 32, 3)) * 40))
    )


def test_single_frame_motion_only_response_is_rejected():
    from audio_narration.domain import Observation
    from audio_narration.vision.vlm import enforce_temporal_evidence

    with pytest.raises(ValueError):
        enforce_temporal_evidence(Observation(0, "A person walked away."), False)


def test_compact_response_preserves_structured_temporal_fields():
    observation = parse_observation(
        '{"s":"A person lifts a cup.", "f":["A desk"], '
        '"a":["lifting a cup"], "c":["The cup was lifted."], '
        '"u":true}',
        1,
    )
    assert observation.facts == ("A desk",)
    assert observation.actions == ("lifting a cup",)
    assert observation.changes == ("The cup was lifted.",)
    assert observation.uncertain


def test_fast_caption_rejects_markup_instead_of_speaking_it():
    from audio_narration.vision.vlm import validate_caption

    assert validate_caption("A person appears to be using a laptop.")
    for invalid in ["", "{}", "```json", "1234", "x" * 501]:
        with pytest.raises(ValueError):
            validate_caption(invalid)
