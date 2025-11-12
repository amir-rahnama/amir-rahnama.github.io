console.log("Typing script loaded ✅");

(function () {
    const titleEl =
        document.querySelector(".profile_inner > h1") ||
        document.querySelector(".home-info h1");
    const descEl = document.querySelector(".profile_inner > span");

    if (!titleEl || !descEl) return;

    /* ---------------- Name typing ---------------- */
    const nameText = titleEl.textContent.trim();
    titleEl.textContent = "";
    titleEl.classList.add("typing-caret");

    let nameIdx = 0;
    const speed = 100; // ms per char (faster)

    function typeNextChar() {
        titleEl.textContent += nameText[nameIdx];
        nameIdx += 1;
        if (nameIdx === nameText.length) {
            titleEl.classList.remove("typing-caret");
            clearInterval(nameTimer);
            // start description typing
            startDescriptionTyping();
        }
    }
    const nameTimer = setInterval(typeNextChar, speed);

    /* --------------- Description typing -------------- */
    const lines = [
        "Senior ML Engineer, All Ears 2025-Now",
        "Senior ML Consultant, Capgemini 2024-2025",
        "ML. PhD, KTH  2019-2024",
        "Research Intern, Spotify 2020"
    ];

    function startDescriptionTyping() {
        descEl.textContent = ""; // clear existing
        let lineIdx = 0;

        function typeLine() {
            if (lineIdx >= lines.length) {
                // done – trigger emoji rain then background gradient
                startEmojiRain();
                document.body.classList.add("bg-animate");
                return;
            }
            const currentLine = lines[lineIdx];
            let charIdx = 0;
            const lineSpan = document.createElement("span");
            lineSpan.classList.add("typing-caret");
            descEl.appendChild(lineSpan);

            const charTimer = setInterval(() => {
                lineSpan.textContent += currentLine[charIdx];
                charIdx += 1;
                if (charIdx === currentLine.length) {
                    clearInterval(charTimer);
                    lineSpan.classList.remove("typing-caret");
                    descEl.appendChild(document.createElement("br"));
                    lineIdx += 1;
                    setTimeout(typeLine, 400); // small pause between lines
                }
            }, speed);
        }

        typeLine();
    }

    /* --------------- Emoji rain -------------- */
    function startEmojiRain() {
        const emojis = ["🎉"];
        const count = 30;
        for (let i = 0; i < count; i++) {
            const span = document.createElement("span");
            span.classList.add("emoji");
            span.textContent = emojis[Math.floor(Math.random() * emojis.length)];
            span.style.left = Math.random() * 100 + "vw";
            span.style.animationDelay = Math.random() * 1 + "s";
            span.style.animationDuration = 2 + Math.random() * 2 + "s";
            document.body.appendChild(span);
            span.addEventListener("animationend", () => span.remove());
        }
    }
})();
