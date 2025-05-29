const paper = {
  metadata: {
    title: "This is a title",
    description: "This is a description",
    keywords: ["This", "Is", "A", "Keyword"],
  },
  authors: [
    {
      name: "John Doe",
      affiliation: {
        name: "University of Example",
        number: "1",
      },
      links: {
        website: "https://johndoe.com",
        google_scholar: "https://scholar.google.com/xxx",
      },
    },
    {
      name: "Jane Smith",
      affiliation: {
        name: "Another University",
        number: "2",
      },
      links: {
        website: "https://janesmith.com",
        google_scholar: "https://scholar.google.com/yyy",
      },
    },
  ],
  links: {
    paper: "https://example.com/paper.pdf",
    arxiv: "https://arxiv.org/abs/xxx",
    video: "https://youtube.com/xxx",
    code: "https://github.com/xxx",
    dataset: "https://example.com/dataset",
  },
  content: {
    abstract:
      "This is the abstract of the paper. It provides a brief overview of the research and its significance.",
    contributions: [
      "First key contribution",
      "Second key contribution",
      "Third key contribution",
    ],
    sections: {
      architecture: {
        image: "./static/images/sample.jpg",
        text: "The architecture consists of...",
      },
      experiments: {
        image: "./static/images/sample.jpg",
        text: "Our experiments show that...",
      },
      case_study: {
        image: "./static/images/sample.jpg",
        text: "In this case study, we demonstrate...",
      },
    },
    conclusion: "In conclusion, our work demonstrates...",
  },
};
