$(document).ready(function () {
  // Register Handlebars helper for equality comparison
  Handlebars.registerHelper('eq', function (a, b) {
    return a === b;
  });

  // Get the template source
  const source = $("#paper-template").html();

  // Compile the template
  const template = Handlebars.compile(source);

  // Render the template with the paper data
  const html = template({ paper: paper });

  // Insert the rendered HTML into the page
  $("#content").html(html);

  // Set the document title
  document.title = paper.metadata.title;

  // Initialize navbar burger functionality after rendering
  $(".navbar-burger").click(function () {
    // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");
  });
});
