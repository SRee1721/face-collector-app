/*navigator.mediaDevices
  .getUserMedia({ video: true })
  .then((stream) => {
    document.getElementById("video").srcObject = stream;
  })
  .catch((err) => {
    console.error("Camera access error:", err);
  });*/

document
  .getElementById("infoForm")
  .addEventListener("submit", async function (e) {
    e.preventDefault();
    const name = document.getElementById("name").value;
    const role = document.getElementById("role").value;

    const response = await fetch("/start-face-collection", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, role }),
    });

    const result = await response.json();
    alert(result.message);
  });
