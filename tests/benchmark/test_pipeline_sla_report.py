from dataclasses import dataclass


@dataclass(slots=True)
class SlaReport:
    parse_chunk_p95_s: float
    retrieval_p95_s: float
    response_p95_s: float

    def passes(self) -> bool:
        return (
            self.parse_chunk_p95_s <= 45.0
            and self.retrieval_p95_s <= 2.5
            and self.response_p95_s <= 8.0
        )



def test_pipeline_sla_report_thresholds() -> None:
    report = SlaReport(parse_chunk_p95_s=0.2, retrieval_p95_s=0.1, response_p95_s=0.1)

    assert report.passes()
