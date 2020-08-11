$(() => {
  var els = document.getElementsByTagName("dt");
  for (var i in els) {
    els[i].innerHTML =
      "<div class='api-docs-inner'>" + els[i].innerHTML + "</div>";
  }
});
