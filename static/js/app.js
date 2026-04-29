const analyzeBtn = document.getElementById("analyzeBtn");

if (analyzeBtn) {
    analyzeBtn.addEventListener("click", async () => {
        const original = document.getElementById("original").value;
        const transformed = document.getElementById("transformed").value;

        try {
            const response = await fetch("/analyze/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    original_path: original,
                    transformed_path: transformed
                })
            });

            const data = await response.json();

            if (!data.success) {
                document.getElementById("metricsResult").innerHTML =
                    `<p style="color:red;">${data.message}</p>`;
                return;
            }

            const result = data.data;

            document.getElementById("metricsResult").innerHTML = `
                <p><strong>MSE:</strong> ${result.metrics.mse}</p>
                <p><strong>PSNR:</strong> ${result.metrics.psnr}</p>
                <p><strong>SSIM:</strong> ${result.metrics.ssim}</p>
                <p><strong>RMSE:</strong> ${result.metrics.rmse}</p>
                <p><strong>Correlation:</strong> ${result.metrics.correlation}</p>
            `;

            document.getElementById("energyResult").innerHTML = `
                <p><strong>Original Energy:</strong> ${result.energy.original}</p>
                <p><strong>Transformed Energy:</strong> ${result.energy.transformed}</p>
                <p><strong>Original Ratio:</strong> ${result.energy.original_ratio}</p>
                <p><strong>Transformed Ratio:</strong> ${result.energy.transformed_ratio}</p>
            `;

            document.getElementById("originalSpectrum").src =
    "/static/results/original_spectrum.png?t=" + new Date().getTime();

document.getElementById("transformedSpectrum").src =
    "/static/results/transformed_spectrum.png?t=" + new Date().getTime();

        } catch (error) {
            document.getElementById("metricsResult").innerHTML =
                `<p style="color:red;">Request failed: ${error}</p>`;
        }
    });
}