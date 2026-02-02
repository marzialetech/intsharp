/**
 * Doc animated GIFs with play/pause/restart controls.
 * Add class="doc-animated-gif" and data-frames, data-fps, data-pause (optional, default 3) to img.
 * The script automatically wraps each GIF with controls.
 */
(function() {
    // CSS for controls
    var style = document.createElement('style');
    style.textContent = `
        .gif-container {
            display: inline-block;
            position: relative;
        }
        .gif-controls {
            display: flex;
            justify-content: center;
            gap: 8px;
            margin-top: 8px;
            padding: 6px;
            background: #f5f5f5;
            border-radius: 4px;
        }
        .gif-controls button {
            padding: 4px 12px;
            font-size: 13px;
            cursor: pointer;
            border: 1px solid #ccc;
            border-radius: 4px;
            background: #fff;
            transition: background 0.2s;
        }
        .gif-controls button:hover {
            background: #e8e8e8;
        }
        .gif-controls button.active {
            background: #ddd;
            font-weight: bold;
        }
        .gif-controls button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
    `;
    document.head.appendChild(style);

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
            
            // State
            var intervalId = null;
            var isPlaying = true;
            
            // Create container
            var container = document.createElement('div');
            container.className = 'gif-container';
            img.parentNode.insertBefore(container, img);
            container.appendChild(img);
            
            // Create controls
            var controls = document.createElement('div');
            controls.className = 'gif-controls';
            
            var playBtn = document.createElement('button');
            playBtn.textContent = 'Pause';
            playBtn.className = 'active';
            
            var restartBtn = document.createElement('button');
            restartBtn.textContent = 'Restart';
            
            controls.appendChild(playBtn);
            controls.appendChild(restartBtn);
            container.appendChild(controls);
            
            function restart() {
                img.src = baseSrc + '?t=' + Date.now();
            }
            
            function startLoop() {
                if (intervalId) clearInterval(intervalId);
                restart();
                intervalId = setInterval(restart, cycleSec * 1000);
                isPlaying = true;
                playBtn.textContent = 'Pause';
                playBtn.className = 'active';
            }
            
            function stopLoop() {
                if (intervalId) {
                    clearInterval(intervalId);
                    intervalId = null;
                }
                isPlaying = false;
                playBtn.textContent = 'Play';
                playBtn.className = '';
            }
            
            playBtn.addEventListener('click', function() {
                if (isPlaying) {
                    stopLoop();
                } else {
                    startLoop();
                }
            });
            
            restartBtn.addEventListener('click', function() {
                restart();
                if (!isPlaying) {
                    startLoop();
                }
            });
            
            // Start auto-loop
            intervalId = setInterval(restart, cycleSec * 1000);
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initGifLoop);
    } else {
        initGifLoop();
    }
})();
