const {
  mean,
  mae,
  r2,
  expectedImprovement,
} = require("../utils/model-utils");

describe("model-utils", () => {
  describe("mean", () => {
    test("calculates the mean of an array of numbers", () => {
      expect(mean([1, 2, 3, 4, 5])).toBe(3);
    });

    test("returns 0 for an empty array", () => {
      expect(mean([])).toBe(0);
    });
  });

  describe("mae", () => {
    test("calculates mean absolute error", () => {
      expect(mae([10, 20], [12, 18])).toBe(2);
    });

    test("returns 0 for empty arrays", () => {
      expect(mae([], [])).toBe(0);
    });
  });

  describe("r2", () => {
    test("calculates R-squared", () => {
      const yTrue = [1, 2, 3, 4, 5];
      const yPred = [1, 2, 3, 4, 5];
      expect(r2(yTrue, yPred)).toBe(1);
    });

    test("returns 1 if ssTot is 0", () => {
      expect(r2([1, 1], [1, 1])).toBe(1);
    });
  });

  describe("expectedImprovement", () => {
    test("returns 0 if sd is 0", () => {
      expect(expectedImprovement(10, 0, 5)).toBe(0);
    });

    test("returns a positive value when mu > bestSoFar", () => {
      const ei = expectedImprovement(10, 1, 5);
      expect(ei).toBeGreaterThan(0);
    });

    test("returns a positive value even when mu < bestSoFar if sd is large", () => {
      const ei = expectedImprovement(4, 2, 5);
      expect(ei).toBeGreaterThan(0);
    });
  });
});
