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

            // --- 1. SONUÇLARI EKRANA YAZDIR ---
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

            // --- 2. SPEKTRUM GÖRSELLERİNİ GÜNCELLE ---
            document.getElementById("originalSpectrum").src =
                "/static/results/original_spectrum.png?t=" + new Date().getTime();

            document.getElementById("transformedSpectrum").src =
                "/static/results/transformed_spectrum.png?t=" + new Date().getTime();

            // --- 3. PDF RAPOR İNDİRME MANTIĞI ---
            const reportBtn = document.getElementById("downloadReportBtn");
            if (reportBtn) {
                // Event listener'ın birden fazla eklenmesini engellemek için butonu klonluyoruz
                const newReportBtn = reportBtn.cloneNode(true);
                reportBtn.parentNode.replaceChild(newReportBtn, reportBtn);

                newReportBtn.addEventListener("click", (e) => {
                    e.preventDefault();

                    // jsPDF kütüphanesini başlat
                    const { jsPDF } = window.jspdf;
                    const doc = new jsPDF();

                    // PDF Tasarımı
                    doc.setFont("helvetica", "bold");
                    doc.setFontSize(20);
                    doc.setTextColor(70, 70, 229); 
                    doc.text("DIAGNOSTIC REPORT", 20, 25);

                    doc.setFontSize(10);
                    doc.setTextColor(100);
                    doc.setFont("helvetica", "normal");
                    doc.text(`Generated on: ${new Date().toLocaleString()}`, 20, 35);
                    doc.line(20, 40, 190, 40); 

                    // Metrikler
                    doc.setFontSize(14);
                    doc.setTextColor(42, 42, 122);
                    doc.setFont("helvetica", "bold");
                    doc.text("METRICS ANALYSIS", 20, 55);

                    doc.setFontSize(12);
                    doc.setFont("helvetica", "normal");
                    doc.setTextColor(0);
                    doc.text(`- MSE: ${result.metrics.mse}`, 25, 65);
                    doc.text(`- PSNR: ${result.metrics.psnr}`, 25, 75);
                    doc.text(`- SSIM: ${result.metrics.ssim}`, 25, 85);
                    doc.text(`- RMSE: ${result.metrics.rmse}`, 25, 95);
                    doc.text(`- Correlation: ${result.metrics.correlation}`, 25, 105);

                    // Enerji Dağılımı
                    doc.setFontSize(14);
                    doc.setFont("helvetica", "bold");
                    doc.setTextColor(42, 42, 122);
                    doc.text("ENERGY DISTRIBUTION", 20, 125);

                    doc.setFontSize(12);
                    doc.setFont("helvetica", "normal");
                    doc.setTextColor(0);
                    doc.text(`- Original Energy: ${result.energy.original}`, 25, 135);
                    doc.text(`- Transformed Energy: ${result.energy.transformed}`, 25, 145);
                    doc.text(`- Original Ratio: ${result.energy.original_ratio}`, 25, 155);
                    doc.text(`- Transformed Ratio: ${result.energy.transformed_ratio}`, 25, 165);

                    doc.setFontSize(10);
                    doc.setTextColor(150);
                    doc.text("Facial Image Analysis Studio - Powered by AI", 20, 280);

                    // PDF'i İndir
                    doc.save("Diagnostic_Report.pdf");
                });
            }

        } catch (error) {
            console.error("Analysis Error:", error);
            document.getElementById("metricsResult").innerHTML =
                `<p style="color:red;">Request failed: ${error.message}</p>`;
        }
    });
}