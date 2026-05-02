import js from "@eslint/js";

export default [
    js.configs.recommended,
    {
        languageOptions: {
            globals: {
                // Node.js
                require: "readonly",
                module: "readonly",
                process: "readonly",
                __dirname: "readonly",
                console: "readonly",
                // Jest
                test: "readonly",
                expect: "readonly",
                describe: "readonly",
                beforeEach: "readonly",
                afterEach: "readonly",
            },
            ecmaVersion: "latest",
            sourceType: "module",
        },
        rules: {
            "no-unused-vars": "warn",
            "no-undef": "error",
        },
    },
];
