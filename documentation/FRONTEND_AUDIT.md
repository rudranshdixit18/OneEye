# OneEye Frontend Audit

## Design system preserved

- Dark tactical surveillance theme using `#06090f`, `#0a0f19`, and `#141d2f` surfaces.
- Sky blue for system telemetry, crimson for critical events, amber for sanitation/safety, and emerald for healthy state.
- Plus Jakarta Sans for the interface and JetBrains Mono for telemetry.
- CCTV HUD, scanlines, reticle, status badges, cards, timeline, toast notifications, and responsive breakpoints are retained.

## Existing sections

1. Dashboard: camera HUD, four camera selectors, counters, filters, and recent detections.
2. Live Feed: start, stop, recording, quality, and alert-volume controls.
3. Detections: searchable detection registry.
4. Analytics: 24-hour visualization, summary cards, and timeline.
5. Settings: notifications and AI thresholds.

## Baseline gaps found

- All incident records and analytics were hard-coded demonstration values.
- The JavaScript made no HTTP, SSE, or WebSocket request.
- Camera controls changed text only and did not control a camera.
- The frontend represented fights, garbage, and people but not fallen persons, phones, or accidents.
- Only fight and garbage thresholds existed.
- Backend status and inference latency were static text.

## Integration applied

- Preserved the original three frontend files and visual language.
- Connected health, events, analytics, settings, camera lifecycle, recording, and MJPEG streaming to `/api/v1`.
- Added fallen-person, mobile-phone, and accident cards, filters, icons, colors, and thresholds using the existing component patterns.
- Kept the original responsive rules and added only narrowly scoped stream/event styles.

