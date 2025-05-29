const paper = {
  metadata: {
    title: "This is a title",
    description: "This is a description",
    keywords: ["This", "Is", "A", "Keyword"],
  },
  navbar: {
    home_link: "https://example.com",
    more_research: [
      {
        name: "Research 1",
        link: "https://example.com/research-1",
      },
      {
        name: "Research 2",
        link: "https://example.com/research-2",
      },
    ],
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
  link_items: [
    {
      name: "Paper",
      link: "https://example.com/paper.pdf",
      icon: "fas fa-file-pdf",
    },
    {
      name: "Arxiv",
      link: "https://arxiv.org/abs/xxx",
      icon: "ai ai-arxiv",
    },
    {
      name: "Video",
      link: "https://youtube.com/xxx",
      icon: "fab fa-youtube",
    },
    {
      name: "Code",
      link: "https://github.com/xxx",
      icon: "fab fa-github",
    },
    {
      name: "Dataset",
      link: "https://example.com/dataset",
      icon: "fa-solid fa-database",
    },
  ],
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
  bibtex: `@article{xxx,
    author    = {xxx},
    title     = {xxx},
    journal   = {xxx},
    year      = {xxx},
  }`,
};
