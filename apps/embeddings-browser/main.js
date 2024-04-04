import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.16.1";

const input1 = document.getElementById("input1");
const input2 = document.getElementById("input2");
const generateButton = document.getElementById("generate-button");
const output = document.getElementById("output");

const generateEmbeddings = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2"
);

generateButton.disabled = false;

function dotProduct(a, b) {
  if (a.length !== b.length) {
    throw new Error("Both arguments must have the same length");
  }
  return a.reduce((acc, val, i) => acc + val * b[i], 0);
}

generateButton.addEventListener("click", async () => {
  const output1 = await generateEmbeddings(input1.value, {
    pooling: "mean",
    normalize: true,
  });

  const output2 = await generateEmbeddings(input2.value, {
    pooling: "mean",
    normalize: true,
  });

  const similarity = dotProduct(output1.data, output2.data);

  output.innerText = similarity;
});
