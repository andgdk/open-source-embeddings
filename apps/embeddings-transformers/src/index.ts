import { Tensor, pipeline, dot } from "@xenova/transformers";

const generateEmbeddings = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2"
);

function dotProduct(a: Tensor, b: Tensor): number {
  return dot(Array.from(a.data), Array.from(b.data));
}

const output1 = await generateEmbeddings("That is a very happy person", {
  pooling: "mean",
  normalize: true,
});

const output2 = await generateEmbeddings("That is a happy person", {
  pooling: "mean",
  normalize: true,
});

const similarity = dotProduct(output1, output2);

console.log(similarity);
