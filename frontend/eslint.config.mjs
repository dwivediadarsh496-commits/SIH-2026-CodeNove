import nextVitals from "eslint-config-next/core-web-vitals";
import nextTypescript from "eslint-config-next/typescript";

const eslintConfig = [
  ...nextVitals,
  ...nextTypescript,
  {
    ignores: [".next/**", "node_modules/**"],
  },
  {
    rules: {
      "@next/next/no-img-element": "off",
    },
  },
];

export default eslintConfig;
