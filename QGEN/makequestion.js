document.addEventListener("DOMContentLoaded", () => {
    const printBtn = document.getElementById("printBtn");
    if (printBtn) {
        printBtn.addEventListener("click", printQuestion);
    }
});

function printQuestion() {
    const questionText = document.getElementById("questionInput").value.trim();
    if (!questionText) {
        alert("Please enter a question before printing.");
        return;
    }

    const printWindow = window.open("", "", "width=794,height=1123"); 
    printWindow.document.write(`
        <html>
        <head>
            <title>Print Question</title>
            <style>
                @page { size: A4 portrait; margin: 2cm; }
                body { font-family: Arial, sans-serif; text-align: center; padding: 20px; }
                h1 { font-size: 24px; text-decoration: underline; }
                p { font-size: 18px; white-space: pre-wrap; text-align: left; margin: 20px; }
            </style>
        </head>
        <body>
            <h1>Question</h1>
            <p>${questionText}</p>
            <script>
                window.onload = function() { window.print(); }
            <\/script>
        </body>
        </html>
    `);
    printWindow.document.close();
}
