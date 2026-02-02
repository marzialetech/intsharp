// Load KaTeX and render LaTeX in the document (\[ \] display, \( \) inline)
(function() {
    var link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css';
    link.crossOrigin = 'anonymous';
    document.head.appendChild(link);

    var script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js';
    script.crossOrigin = 'anonymous';
    script.defer = true;
    script.onload = function() {
        var ar = document.createElement('script');
        ar.src = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js';
        ar.crossOrigin = 'anonymous';
        ar.defer = true;
        ar.onload = function() {
            if (typeof renderMathInElement === 'function') {
                renderMathInElement(document.body, {
                    delimiters: [
                        { left: '\\[', right: '\\]', display: true },
                        { left: '\\(', right: '\\)', display: false }
                    ],
                    throwOnError: false
                });
            }
        };
        document.head.appendChild(ar);
    };
    document.head.appendChild(script);
})();
