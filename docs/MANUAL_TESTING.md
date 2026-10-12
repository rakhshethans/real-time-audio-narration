# Manual activity and accessibility evaluation

Use a consenting participant, good lighting, a stationary webcam and no private documents in
view. Run setup before disconnecting networking. Do not record by default. For replay evaluation,
record only with participants' explicit consent using a separate recording tool, and keep videos
out of Git. Never use this prototype to assess safety or navigate a hazardous environment.

Test both modes with identical camera angle, confidence, resolution, detail and analysis spacing.
Warm the models first; record hardware, package versions, chosen device and model revisions.
Run each scenario at least three times. Note missed actions, invented events, uncertainty,
repetition, track-ID continuity, and approximate event-to-audio delay. Keep a human event log to
measure action onset; the UI's frame-to-audio proxy is not ground truth.

| Scenario | Procedure | Inspect |
| --- | --- | --- |
| Entering | Start with an empty chair; person enters and remains visible | New arrival described; no name or sensitive trait inference |
| Leaving | Person leaves the view after a stable scene | Visible departure described; occlusion not treated confidently as leaving |
| Bottle pickup | Put bottle on desk, reach and lift it over several seconds | Person-object interaction; detection alone does not assert pickup |
| Drinking | Lift bottle to mouth, tilt and lower | Drinking described cautiously if ambiguous; distinguish holding from drinking |
| Sit / stand | Stand beside chair, sit, then stand again | Ordered action changes rather than repeating furniture description |
| Object movement | Move a cup across desk, stop, then move it back | New spatial relationship; stable scene then suppressed |
| Static scene | Keep scene still for at least two minutes | First description, no repeated events, only occasional summaries |

Also test a single still frame, partial occlusion, dim light, fast motion and camera movement.
Compare output with visual evidence, not with the text of an expected scripted description.

Reliability checks: disconnect camera while running; mute during synthesis/playback; pause and
resume rapidly while Qwen is busy; stop while models load; stop while audio plays; restart the
session; choose an invalid camera index; unplug the audio device; remove a model weight and
confirm a useful error. Confirm preview continues during inference and no old descriptions
are heard after pause or stop. Observe CPU/RAM for ten minutes to check bounded memory.

Accessibility checks: operate controls with Tab, Space, keyboard mnemonics and Escape; verify
visible focus; try a Windows screen reader; ensure status/history labels are useful and error
messages are discoverable. User testing is needed before claiming accessible usability.

Report precision/recall for annotated events, unsupported-claim rate, repeated-event rate,
mean and percentile delays, VLM calls/minute and CPU/RAM. Do not equate unit-test success with
activity recognition accuracy. See VALIDATION.md for checks already conducted.
