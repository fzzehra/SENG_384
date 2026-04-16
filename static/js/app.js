const analyzeBtn = document.getElementById("analyzeBtn");

if (analyzeBtn) {
    analyzeBtn.addEventListener("click", async () => {
        const original = document.getElementById("original").value;
        const transformed = document.getElementById("transformed").value;

        try {
            const response = await fetch("http://127.0.0.1:5000/analyze", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    original: original,
                    transformed: transformed
                })
            });

            const data = await response.json();

            if (data.error) {
                document.getElementById("metricsResult").innerHTML =
                    `<p style="color:red;">${data.error}</p>`;
                return;
            }

            document.getElementById("metricsResult").innerHTML = `
                <p><strong>MSE:</strong> ${data.metrics.mse}</p>
                <p><strong>PSNR:</strong> ${data.metrics.psnr}</p>
                <p><strong>SSIM:</strong> ${data.metrics.ssim}</p>
            `;

            document.getElementById("energyResult").innerHTML = `
                <p><strong>Original Energy:</strong> ${data.energy.original}</p>
                <p><strong>Transformed Energy:</strong> ${data.energy.transformed}</p>
                <p><strong>Original Low Energy:</strong> ${data.energy.original_low}</p>
                <p><strong>Original High Energy:</strong> ${data.energy.original_high}</p>
                <p><strong>Transformed Low Energy:</strong> ${data.energy.transformed_low}</p>
                <p><strong>Transformed High Energy:</strong> ${data.energy.transformed_high}</p>
                <p><strong>Original Ratio:</strong> ${data.energy.original_ratio}</p>
                <p><strong>Transformed Ratio:</strong> ${data.energy.transformed_ratio}</p>
                <p><strong>Ratio Difference:</strong> ${data.energy.ratio_difference}</p>
            `;

            document.getElementById("originalSpectrum").src =
                data.outputs.original_spectrum + "?t=" + new Date().getTime();

            document.getElementById("transformedSpectrum").src =
                data.outputs.transformed_spectrum + "?t=" + new Date().getTime();

        } catch (error) {
            document.getElementById("metricsResult").innerHTML =
                `<p style="color:red;">Request failed: ${error}</p>`;
        }
    });
}