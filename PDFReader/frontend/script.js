const API_URL = "http://127.0.0.1:8000";


// =====================================================
// UPLOAD PDF
// =====================================================

async function uploadPDF() {

    const fileInput = document.getElementById("pdfFile");
    const uploadButton = document.getElementById("uploadButton");
    const status = document.getElementById("uploadStatus");

    if (!fileInput.files.length) {

        status.innerText = "Please select a PDF file.";

        return;
    }

    const file = fileInput.files[0];

    if (!file.name.toLowerCase().endsWith(".pdf")) {

        status.innerText = "Only PDF files are allowed.";

        return;
    }

    const formData = new FormData();

    formData.append("file", file);


    uploadButton.disabled = true;

    uploadButton.innerText = "Processing...";

    status.innerText = "Uploading and processing PDF...";


    try {

        const response = await fetch(
            `${API_URL}/upload`,
            {
                method: "POST",
                body: formData
            }
        );


        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail || "Upload failed"
            );
        }


        status.innerText =
            `✅ ${data.message}
            
File: ${data.filename}
Pages: ${data.pages}
Chunks: ${data.chunks}`;


    }
    catch (error) {

        console.error(error);

        status.innerText =
            `❌ ${error.message}`;

    }
    finally {

        uploadButton.disabled = false;

        uploadButton.innerText = "Upload PDF";
    }
}



// =====================================================
// ASK QUESTION
// =====================================================

async function askQuestion() {

    const questionInput =
        document.getElementById("question");

    const answer =
        document.getElementById("answer");

    const askButton =
        document.getElementById("askButton");


    const question =
        questionInput.value.trim();


    if (!question) {

        answer.innerText =
            "Please enter a question.";

        return;
    }


    askButton.disabled = true;

    askButton.innerText = "Thinking...";

    answer.innerText =
        "Searching the PDF...";


    try {

        const response = await fetch(
            `${API_URL}/ask`,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    question: question
                })
            }
        );


        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail || "Question failed"
            );
        }


        answer.innerText =
            data.answer;

    }
    catch (error) {

        console.error(error);

        answer.innerText =
            `❌ ${error.message}`;

    }
    finally {

        askButton.disabled = false;

        askButton.innerText =
            "Ask Question";
    }
}