function logout() {
  window.location.href = "index.html";
}

async function analyze() {
  const file = document.getElementById("imageInput").files[0];
  const result = document.getElementById("result");

  if (!file) {
    result.innerText = "Please upload an image first.";
    return;
  }

  result.innerText = "Analyzing image...";

  const formData = new FormData();
  formData.append("image", file);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("Request failed");
    }

    const data = await response.json();
    result.innerText = data.result || "Analysis completed.";
  } catch (error) {
    result.innerText = "Unable to process the image right now. Please try again.";
  }
}
