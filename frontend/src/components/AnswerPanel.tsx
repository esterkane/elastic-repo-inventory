import type { AnswerResponse, SearchHit } from "../lib/api";
import { formatAnswer } from "../lib/resultFormatter";
import { AnswerSummary } from "./AnswerSummary";
import { EvidenceCard } from "./EvidenceCard";
import { QueryInsights } from "./QueryInsights";
import { SourceNavigator } from "./SourceNavigator";

type AnswerPanelProps = {
  answer: AnswerResponse | null;
  searchHits?: SearchHit[];
  isLoading?: boolean;
};

export function AnswerPanel({ answer, searchHits = [], isLoading = false }: AnswerPanelProps) {
  const hasSynthesizedAnswer = answerHasEvidence(answer);
  const effectiveAnswer = hasSynthesizedAnswer ? answer : answerFromSearchHits(searchHits);

  if (!effectiveAnswer && !isLoading) {
    return (
      <section className="answer-explorer answer-explorer-empty" aria-label="Answer explorer">
        <div className="answer-empty-card">
          <p className="eyebrow">Release briefing</p>
          <h2>Pick a version range and topic</h2>
          <p>
            Search will summarize what changed, why it matters, what to inspect, and the best Elasticsearch source to open first.
          </p>
        </div>
      </section>
    );
  }

  const model = formatAnswer(effectiveAnswer);
  const fallbackNotice = !hasSynthesizedAnswer && !isLoading ? effectiveAnswer?.explanation ?? null : null;

  return (
    <section className="answer-explorer" aria-label="Answer explorer">
      <AnswerSummary model={model} isLoading={isLoading} fallbackNotice={fallbackNotice} />

      <section className="evidence-panel" aria-labelledby="evidence-heading">
        <div className="panel-heading">
          <h2 id="evidence-heading">Evidence</h2>
        </div>
        {model.primaryEvidence ? (
          <>
            <EvidenceCard evidence={model.primaryEvidence} primary />
            {model.supportingEvidence.length > 0 && (
              <div className="supporting-evidence">
                <p className="result-group-label">More useful excerpts</p>
                {model.supportingEvidence.slice(0, 2).map((item) => (
                  <EvidenceCard evidence={item} key={`${item.title}-${item.reader_url}-${item.source_url}`} />
                ))}
              </div>
            )}
            <QueryInsights primaryTitle={model.primaryEvidence.title} />
          </>
        ) : (
          <p className="empty-state">Short excerpts will appear here after the answer endpoint returns release sources.</p>
        )}
      </section>

      <SourceNavigator bestSource={model.bestSource} supportingSources={model.supportingSources} />
    </section>
  );
}

function answerHasEvidence(answer: AnswerResponse | null): answer is AnswerResponse {
  return Boolean(answer?.evidence?.length);
}

function answerFromSearchHits(hits: SearchHit[]): AnswerResponse | null {
  const usefulHits = hits.filter((hit) => hit.snippet && hit.source_url).slice(0, 3);
  if (usefulHits.length === 0) {
    return null;
  }

  const evidence = usefulHits.map((hit, index) => {
    const readerUrl = readerUrlFor(hit);
    return {
      title: hit.title ?? hit.heading_path ?? hit.path ?? "Untitled source",
      heading_path: hit.heading_path,
      repo: hit.repo,
      path: hit.path,
      content_type: hit.content_type,
      license_family: hit.license_family,
      score: hit.score,
      role: index === 0 ? "primary" as const : "supporting" as const,
      claim: hit.snippet ?? "",
      excerpt: hit.snippet ?? "",
      highlight_terms: hit.highlights ?? [],
      reader_url: readerUrl,
      source_url: hit.source_url,
      link_label: hit.repo === "elastic/docs-content" ? "Read documentation" as const : "View source" as const
    };
  });
  const sources = evidence.map((item) => ({
    title: item.title,
    url: item.reader_url,
    link_label: item.link_label,
    repo: item.repo,
    path: item.path,
    heading_path: item.heading_path
  }));

  return {
    summary: usefulHits[0]?.snippet ?? "The search results contain relevant documentation evidence.",
    direct_answer: "The strongest result points to an Elasticsearch source that should be opened first.",
    explanation:
      "The answer endpoint did not return a full synthesis, so the page is using ranked release sources directly. Start with the first source, then use related sources only when they add an example, caveat, or implementation detail.",
    important: "This keeps the briefing tied to concrete Elasticsearch changes while synthesis is unavailable.",
    confidence: usefulHits.length >= 2 ? "medium" : "low",
    evidence,
    links: sources,
    best_source: sources[0],
    supporting_sources: sources.slice(1),
    warnings: [],
    degraded: false
  };
}

function readerUrlFor(hit: SearchHit): string {
  if (hit.repo !== "elastic/docs-content" || !hit.path) {
    return hit.source_url;
  }
  const anchor = hit.source_url.includes("#") ? hit.source_url.slice(hit.source_url.indexOf("#")) : "";
  const docsPath = hit.path.replace(/\.(md|mdx)$/i, "");
  return `https://www.elastic.co/docs/${docsPath}${anchor}`;
}
