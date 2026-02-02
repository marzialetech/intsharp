/**
 * Doc animated GIFs: play once, pause at last frame, then restart.
 * Add class="doc-animated-gif" and data-frames, data-fps, data-pause (optional, default 3) to img.
 */
(function() {
    function initGifLoop() {
        var imgs = document.querySelectorAll('img.doc-animated-gif');
        imgs.forEach(function(img) {
            var frames = parseInt(img.getAttribute('data-frames'), 10);
            var fps = parseInt(img.getAttribute('data-fps'), 10) || 10;
            var pause = parseFloat(img.getAttribute('data-pause'), 10);
            if (isNaN(pause)) pause = 3;
            if (isNaN(frames) || isNaN(fps) || frames <= 0 || fps <= 0) return;
            var durationSec = frames / fps;
            var cycleSec = durationSec + pause;
            var baseSrc = img.src.split('?')[0];

            function restart() {
                img.src = baseSrc + '?t=' + Date.now();
            }

            setInterval(restart, cycleSec * 1000);
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initGifLoop);
    } else {
        initGifLoop();
    }
})();
