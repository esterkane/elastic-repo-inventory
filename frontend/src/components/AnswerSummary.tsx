import { ArrowRight, MapPin } from "lucide-react";
import type { AnswerViewModel } from "../lib/resultFormatter";
import { ConfidenceBadge } from "./ConfidenceBadge";
import { SourceMetadata } from "./SourceMetadata";

type AnswerSummaryProps = {
  model: AnswerViewModel;
  isLoading?: boolean;
  fallbackNotice?: string | null;
};

export function AnswerSummary({ model, isLoading = false, fallbackNotice = null }: AnswerSummaryProps) {
  return (
    <section className="answer-summary-panel" aria-labelledby="answer-heading">
      <div className="answer-summary-panel__top">
        <div>
          <p className="eyebrow">Start here</p>
          <h2 id="answer-heading">Answer</h2>
        </div>
        <ConfidenceBadge confidence={model.confidence} />
      </div>
      <p className="answer-summary">
        {isLoading ? "Building a grounded answer from the strongest evidence." : model.directAnswer}
      </p>
      {model.whatNew.length > 0 && (
        <div className="insight-block insight-block-release">
          <h3>What's new</h3>
          <ul>
            {model.whatNew.slice(0, 5).map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      )}
      <div className="explain-block">
        <h3>Why it matters</h3>
        {fallbackNotice && <p className="fallback-notice">{fallbackNotice}</p>}
        <p>{model.explanation}</p>
        <p>{model.important}</p>
      </div>
      {model.whatToNotice.length > 0 && (
        <div className="insight-block insight-block-new">
          <h3>What to look for</h3>
          <ul>
            {model.whatToNotice.slice(0, 4).map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      )}
      <div className="insight-grid">
        {model.bestSource && (
          <div className="insight-block">
            <h3>Best source</h3>
            <SourceMetadata display={model.bestSource.display} includeTitle />
            <a className="best-link" href={model.bestSource.url} target="_blank" rel="noreferrer">
              <MapPin aria-hidden="true" size={15} />
              <span>{model.bestSource.link_label === "Read documentation" ? "Read docs" : "Open source"}</span>
              <ArrowRight aria-hidden="true" size={15} />
            </a>
          </div>
        )}
        <div className="insight-block">
          <h3>Related sources</h3>
          {model.supportingSources.length > 0 ? (
            <ul>
              {model.supportingSources.slice(0, 3).map((source) => (
                <li key={`${source.title}-${source.url}`}>{source.display.title}</li>
              ))}
            </ul>
          ) : (
            <p>{model.supportingContext}</p>
          )}
        </div>
      </div>
    </section>
  );
}
