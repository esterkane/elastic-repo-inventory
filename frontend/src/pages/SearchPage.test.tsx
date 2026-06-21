import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { SearchPage } from "./SearchPage";
import { hybridRetrievalAnswer, hybridRetrievalSearch } from "../test/fixtures";
import { answer, search } from "../lib/api";

vi.mock("../lib/api", async () => {
  const actual = await vi.importActual<typeof import("../lib/api")>("../lib/api");
  return {
    ...actual,
    answer: vi.fn(),
    ingestRepo: vi.fn(),
    search: vi.fn()
  };
});

describe("SearchPage", () => {
  beforeEach(() => {
    vi.mocked(search).mockResolvedValue(hybridRetrievalSearch);
    vi.mocked(answer).mockResolvedValue(hybridRetrievalAnswer);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("does not show internal improvement suggestions in the default UI", () => {
    render(<SearchPage />);

    expect(screen.queryByText(/Improvement Suggestions/i)).not.toBeInTheDocument();
  });

  it("keeps secondary controls collapsed by default", () => {
    render(<SearchPage />);

    const advanced = screen.getByText("Secondary filters").closest("details");
    expect(advanced).not.toHaveAttribute("open");
  });

  it("submits search and answer requests without calling analyze", async () => {
    render(<SearchPage />);

    fireEvent.click(screen.getByRole("button", { name: /Search/i }));

    await waitFor(() => {
      expect(search).toHaveBeenCalledWith(expect.objectContaining({
        query: expect.stringContaining("vector search memory improvements"),
        topic: "vector_search",
        version_range: { from: "9.0", to: "9.2" },
        time_range: "latest"
      }));
      expect(answer).toHaveBeenCalledWith(expect.objectContaining({ query: expect.stringContaining("latest 9.x 8.x") }));
    });
    expect(screen.queryByText(/Improvement Suggestions/i)).not.toBeInTheDocument();
    expect(screen.getByText(/Use hybrid retrieval to gather candidates/i)).toBeInTheDocument();
  });

  it("shows version-aware release intelligence controls", () => {
    render(<SearchPage />);

    expect(screen.getByLabelText(/Topic/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Version from/i)).toHaveValue("9.0");
    expect(screen.getByLabelText(/Version to/i)).toHaveValue("9.2");
    expect(screen.getByLabelText(/Time range/i)).toHaveValue("latest");
  });

  it("renders the answer region before the results region", async () => {
    render(<SearchPage />);

    fireEvent.click(screen.getByRole("button", { name: /Search/i }));

    const answerHeading = await screen.findByRole("heading", { name: "Answer" });
    const resultsHeading = screen.getByRole("heading", { name: "Sources" });

    expect(answerHeading.compareDocumentPosition(resultsHeading) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });
});
