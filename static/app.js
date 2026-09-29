// ============================================================
// DROWSENSE AI
// Frontend Dashboard Controller
// ============================================================

const earHistory = [];
const marHistory = [];

const MAX_HISTORY = 80;


// ============================================================
// ELEMENTS
// ============================================================

const statusBox =
    document.getElementById("statusBox");

const statusMain =
    document.getElementById("statusMain");

const statusIcon =
    document.getElementById("statusIcon");

const score =
    document.getElementById("score");

const scoreFill =
    document.getElementById("scoreFill");

const ear =
    document.getElementById("ear");

const mar =
    document.getElementById("mar");

const threshold =
    document.getElementById("threshold");

const fps =
    document.getElementById("fps");

const pitch =
    document.getElementById("posepitch");

const yaw =
    document.getElementById("poseyaw");

const roll =
    document.getElementById("poseroll");

const face =
    document.getElementById("face");

const calibration =
    document.getElementById("calibration");

const headerFps =
    document.getElementById("headerFps");

const cameraTime =
    document.getElementById("cameraTime");


// ============================================================
// CANVAS
// ============================================================

const earCanvas =
    document.getElementById("earChart");

const marCanvas =
    document.getElementById("marChart");

const earCtx =
    earCanvas.getContext("2d");

const marCtx =
    marCanvas.getContext("2d");


// ============================================================
// UPDATE CLOCK
// ============================================================

function updateClock() {

    const now = new Date();

    cameraTime.textContent =
        now.toLocaleTimeString();

}

setInterval(updateClock, 1000);

updateClock();


// ============================================================
// STATUS
// ============================================================

function updateStatus(data) {

    const currentStatus =
        data.status || "STARTING";


    statusBox.className =
        "status-box";


    if (
        currentStatus ===
        "DROWSINESS DETECTED"
    ) {

        statusBox.classList.add(
            "danger"
        );

        statusMain.textContent =
            "DROWSINESS DETECTED";

        statusIcon.textContent =
            "!";

    }

    else if (
        currentStatus ===
        "CALIBRATING"
    ) {

        statusBox.classList.add(
            "warning"
        );

        statusMain.textContent =
            "CALIBRATING";

        statusIcon.textContent =
            "C";

    }

    else if (
        currentStatus ===
        "NO FACE DETECTED"
    ) {

        statusBox.classList.add(
            "warning"
        );

        statusMain.textContent =
            "NO FACE DETECTED";

        statusIcon.textContent =
            "?";

    }

    else {

        statusMain.textContent =
            "NORMAL";

        statusIcon.textContent =
            "OK";

    }

}


// ============================================================
// UPDATE METRICS
// ============================================================

function updateMetrics(data) {

    ear.textContent =
        Number(data.ear).toFixed(3);

    mar.textContent =
        Number(data.mar).toFixed(3);

    fps.textContent =
        Number(data.fps).toFixed(1);

    headerFps.textContent =
        Number(data.fps).toFixed(1);

    pitch.textContent =
        Number(data.pitch).toFixed(1) + "°";

    yaw.textContent =
        Number(data.yaw).toFixed(1) + "°";

    roll.textContent =
        Number(data.roll).toFixed(1) + "°";


    if (data.ear_threshold !== null) {

        threshold.textContent =
            Number(
                data.ear_threshold
            ).toFixed(3);

    }

    else {

        threshold.textContent =
            "--";

    }


    face.textContent =
        data.face_detected
            ? "DETECTED"
            : "NOT FOUND";


    score.textContent =
        data.score;


    const scoreValue =
        Math.max(
            0,
            Math.min(
                100,
                data.score
            )
        );


    scoreFill.style.width =
        scoreValue + "%";


    if (scoreValue >= 60) {

        scoreFill.style.background =
            "var(--red)";

    }

    else if (scoreValue >= 30) {

        scoreFill.style.background =
            "var(--yellow)";

    }

    else {

        scoreFill.style.background =
            "var(--green)";

    }


    if (
        data.status ===
        "CALIBRATING"
    ) {

        calibration.textContent =
            "CALIBRATING — " +
            data.calibration_remaining +
            "s";

    }

    else if (
        data.ear_threshold !== null
    ) {

        calibration.textContent =
            "COMPLETE";

    }

    else {

        calibration.textContent =
            "WAITING";

    }

}


// ============================================================
// GRAPH HISTORY
// ============================================================

function addHistory(data) {

    earHistory.push(
        Number(data.ear)
    );

    marHistory.push(
        Number(data.mar)
    );


    if (
        earHistory.length >
        MAX_HISTORY
    ) {

        earHistory.shift();

    }


    if (
        marHistory.length >
        MAX_HISTORY
    ) {

        marHistory.shift();

    }

}


// ============================================================
// DRAW GRAPH
// ============================================================

function drawGraph(
    ctx,
    canvas,
    values,
    minValue,
    maxValue,
    thresholdValue
) {

    const width =
        canvas.clientWidth;

    const height =
        canvas.clientHeight;


    if (
        width <= 0 ||
        height <= 0
    ) {

        return;

    }


    const dpr =
        window.devicePixelRatio || 1;


    canvas.width =
        width * dpr;

    canvas.height =
        height * dpr;


    ctx.setTransform(
        dpr,
        0,
        0,
        dpr,
        0,
        0
    );


    ctx.clearRect(
        0,
        0,
        width,
        height
    );


    // Grid

    ctx.strokeStyle =
        "rgba(255,255,255,0.06)";

    ctx.lineWidth = 1;


    for (
        let i = 1;
        i < 4;
        i++
    ) {

        const y =
            (height / 4) * i;

        ctx.beginPath();

        ctx.moveTo(
            0,
            y
        );

        ctx.lineTo(
            width,
            y
        );

        ctx.stroke();

    }


    if (
        values.length < 2
    ) {

        return;

    }


    // Threshold line

    if (
        thresholdValue !== null
    ) {

        const normalized =
            (
                thresholdValue -
                minValue
            ) /
            (
                maxValue -
                minValue
            );


        const y =
            height -
            normalized * height;


        ctx.strokeStyle =
            "rgba(255,59,79,0.55)";

        ctx.setLineDash(
            [5, 5]
        );

        ctx.beginPath();

        ctx.moveTo(
            0,
            y
        );

        ctx.lineTo(
            width,
            y
        );

        ctx.stroke();

        ctx.setLineDash([]);

    }


    // Main graph

    ctx.strokeStyle =
        "#4d8dff";

    ctx.lineWidth = 2;

    ctx.beginPath();


    values.forEach(
        (value, index) => {

            const x =
                (
                    index /
                    (values.length - 1)
                ) * width;


            const normalized =
                (
                    value -
                    minValue
                ) /
                (
                    maxValue -
                    minValue
                );


            const y =
                height -
                Math.max(
                    0,
                    Math.min(
                        1,
                        normalized
                    )
                ) * height;


            if (index === 0) {

                ctx.moveTo(
                    x,
                    y
                );

            }

            else {

                ctx.lineTo(
                    x,
                    y
                );

            }

        }
    );


    ctx.stroke();

}


// ============================================================
// DRAW CHARTS
// ============================================================

function drawCharts(data) {

    drawGraph(
        earCtx,
        earCanvas,
        earHistory,
        0.10,
        0.45,
        data.ear_threshold
    );


    drawGraph(
        marCtx,
        marCanvas,
        marHistory,
        0.0,
        1.0,
        0.65
    );

}


// ============================================================
// GET STATUS FROM FLASK
// ============================================================

async function updateDashboard() {

    try {

        const response =
            await fetch(
                "/api/status",
                {
                    cache: "no-store"
                }
            );


        const data =
            await response.json();


        updateStatus(data);

        updateMetrics(data);

        addHistory(data);

        drawCharts(data);

    }

    catch (error) {

        console.error(
            "Dashboard update failed:",
            error
        );

    }

}


// ============================================================
// RECALIBRATION
// ============================================================

async function recalibrate() {

    try {

        await fetch(
            "/api/recalibrate",
            {
                method: "POST"
            }
        );

    }

    catch (error) {

        console.error(
            "Recalibration failed:",
            error
        );

    }

}


// ============================================================
// REFRESH
// ============================================================

setInterval(
    updateDashboard,
    200
);


updateDashboard();


// ============================================================
// RESPONSIVE GRAPH REDRAW
// ============================================================

window.addEventListener(
    "resize",
    () => {

        updateDashboard();

    }
);