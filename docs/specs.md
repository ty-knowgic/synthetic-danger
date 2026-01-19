# Technical Specifications

Synthetic Danger relies on structured data formats and a predefined safety ontology to ensure consistency and quality in its generated reports.

## Hazard JSON Schema

All generated hazards must adhere to the following JSON schema. This structure ensures that downstream normalization and reporting tools can process the data reliably.

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `hazard_id` | String | Unique identifier (min 6 characters). |
| `category` | Enum | Broad safety category (e.g., Mechanical, Electrical). |
| `subtype` | String | Specific type of failure within the category. |
| `trigger_condition` | String | Initial condition that leads to the hazard. |
| `primary_failure` | String | The core failure event. |
| `propagation_chain` | Array | Sequence of events (min 4 steps) leading to impact. |
| `final_impact` | String | Resulting consequence (harm or damage). |
| `severity` | Enum | `Low`, `Medium`, `High`, `Critical`. |
| `likelihood` | Enum | `Low`, `Medium`, `High`. |
| `detectability` | Enum | `Low`, `Medium`, `High`. |
| `affected_components`| Array | List of system components involved. |
| `task_phase` | String | Active system state or operator task. |

## Hazard Ontology

The following categories and subtypes represent the "thought space" explored by the LLM during hazard identification.

### Mechanical
- Collision, DroppedObject, PinchPoint, WearAndTear, Overload, FastenerFailure.

### Electrical
- PowerLoss, Brownout, EMI, GroundFault, Overcurrent, EStopFailure.

### Software
- StateDesync, UnhandledException, RaceCondition, MemoryLeak, ConfigError, UpdateRegression.

### Perception
- Occlusion, GlareReflection, MotionBlur, LowContrast, Misclassification, DomainShift.

### Human Interaction
- Misuse, UnexpectedEntry, HRIConfusion, PPEViolation, TrainingGap.

### Environment
- TemperatureExtreme, Humidity, Dust, Vibration, PoorLighting, ReflectiveSurfaces.

## Risk Assessment Model

The tool generates draft values for **Severity**, **Likelihood**, and **Detectability**. These are intented to be reviewed and adjusted by safety engineers using a standard Risk Priority Number (RPN) logic in follow-up meetings.
