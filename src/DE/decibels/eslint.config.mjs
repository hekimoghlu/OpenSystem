// @ts-check

import eslint from "@eslint/js";
import prettier from "eslint-plugin-prettier/recommended";
import tseslint from "typescript-eslint";

export default tseslint.config(
    eslint.configs.recommended,
    tseslint.configs.recommendedTypeChecked,
    prettier,
    {
        files: ["**/*.ts", "**/*.tsx"],
        ignores: [
            "types/**/*",
            "gi-types/**/*",
            "**/.eslintrc.js",
            "**/_build/",
        ],
        languageOptions: {
            parserOptions: {
                projectService: true,
                /// @ts-expect-error Node.js >=20.11.0 / >= 21.2.0
                tsconfigRootDir: import.meta.dirname,
            },
        },
        rules: {
            "@typescript-eslint/restrict-template-expressions": ["error", {
                allowNullish: true,
            }],
            "@typescript-eslint/no-unused-vars": ["error", {
                argsIgnorePattern: "^_",
                varsIgnorePattern: "^_",
            }],
            "prettier/prettier": ["error"],
        },
    },
);
