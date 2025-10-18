// This hack comes from NewsFlash but is simplified by Biblioteca
// to avoid having to process incoming content for the WebKitWebView.
// GPLv3+.
//
// https://github.com/workbenchdev/Biblioteca/blob/022ede135351cbff519e456ab38b6624e3938e0b/src/WebView.js#L51

var body = document.body;
var divTop = document.createElement('div');
body.insertBefore(divTop, body.firstChild);
var divBottom = document.createElement('div');
body.insertAdjacentElement('beforeend', divBottom);

window.addEventListener('scroll', on_scroll);

function on_scroll() {
    if (window.scrollY > 0) {
      divTop.classList.add("overshoot-overlay-top");
    } else {
      divTop.classList.remove("overshoot-overlay-top");
    }

    var limit = Math.max(
      document.body.scrollHeight,
      document.body.offsetHeight,
      document.documentElement.clientHeight,
      document.documentElement.scrollHeight,
      document.documentElement.offsetHeight);
    var max_scroll = limit - window.innerHeight;

    if (window.scrollY >= max_scroll) {
      divBottom.classList.remove("overshoot-overlay-bottom");
    } else {
      divBottom.classList.add("overshoot-overlay-bottom");
    }
}
