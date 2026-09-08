# SentinelBMS Knowledge Base

Short reference advisories used by the explanation step. Each section covers
one anomaly type: what causes it, why it matters, and the recommended action.

## cell_voltage_deviation

Anomaly type: cell_voltage_deviation. In a healthy battery pack every cell
holds nearly the same voltage, so a single cell reading far from the pack
average usually means a failing cell, a damaged balancing circuit, or a
spoofed voltage sensor feeding false data to the BMS. This matters because a
weak or misreported cell can be overcharged without the BMS noticing, which
is a known precursor to overheating incidents. Past fleet reviews found that
deviations above 15% during charging preceded pack failures within two
weeks. Recommended action: stop fast charging, reduce load on the vehicle,
and schedule a cell-level inspection before the next long trip.

## firmware_integrity

Anomaly type: firmware_integrity. Every legitimate BMS firmware release is
digitally signed by the manufacturer, and its hash is recorded in a
known-good registry, so a hash mismatch means the running software is not a
recognized release. Modified firmware can disable thermal cutoffs or
overcurrent protections, which is why this is treated as a potential
security compromise rather than a routine glitch. Vehicles running unsigned
firmware have been linked in incident reports to charging systems that kept
operating past safe temperature limits. Recommended action: treat as a
security incident: park the vehicle, disconnect it from any charger, and
have an authorized service center re-flash signed firmware.

## soc_inconsistency

Anomaly type: soc_inconsistency. Because cell voltage rises predictably with
charge level, the BMS can cross-check the state of charge it reports against
what the measured voltages imply. A wide gap between the two usually means a
miscalibrated sensor, a software fault, or a deliberately spoofed report.
This matters because drivers who trust an inflated charge reading can suffer
sudden power loss, including documented cases at highway speeds. Recommended
action: do not rely on the displayed range, keep the battery topped up, and
have the BMS firmware and calibration verified by a service center.
