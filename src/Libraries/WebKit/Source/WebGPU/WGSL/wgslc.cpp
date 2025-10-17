/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 9, 2023.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include "config.h"

#include "AST/ASTStringDumper.h"
#include "WGSL.h"
#include "WGSLShaderModule.h"
#include <wtf/DataLog.h>
#include <wtf/FileSystem.h>
#include <wtf/WTFProcess.h>

static NO_RETURN void printUsageStatement(bool help = false)
{
    fprintf(stderr, "Usage: wgsl [options] <file> <entrypoint>\n");
    fprintf(stderr, "  -h|--help  Prints this help message\n");
    fprintf(stderr, "  --dump-ast-after-checking  Dumps the AST after parsing and checking\n");
    fprintf(stderr, "  --dump-ast-at-end  Dumps the AST after generating code\n");
    fprintf(stderr, "  --dump-generated-code  Dumps the generated Metal code\n");
    fprintf(stderr, "\n");

    exitProcess(help ? EXIT_SUCCESS : EXIT_FAILURE);
}

struct CommandLine {
public:
    CommandLine(int argc, char** argv)
    {
        parseArguments(argc, argv);
    }

    const char* file() const { return m_file; }
    const char* entrypoint() const { return m_entrypoint; }
    bool dumpASTAfterCheck() const { return m_dumpASTAfterCheck; }
    bool dumpASTAtEnd() const { return m_dumpASTAtEnd; }
    bool dumpGeneratedCode() const { return m_dumpGeneratedCode; }

private:
    void parseArguments(int, char**);

    const char* m_file { nullptr };
    const char* m_entrypoint { nullptr };
    bool m_dumpASTAfterCheck { false };
    bool m_dumpASTAtEnd { false };
    bool m_dumpGeneratedCode { false };
};

void CommandLine::parseArguments(int argc, char** argv)
{
    for (int i = 1; i < argc; ++i) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
        const char* arg = argv[i];
#pragma clang diagnostic pop
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help"))
            printUsageStatement(true);

        if (!strcmp(arg, "--dump-ast-after-checking")) {
            m_dumpASTAfterCheck = true;
            continue;
        }

        if (!strcmp(arg, "--dump-ast-at-end")) {
            m_dumpASTAtEnd = true;
            continue;
        }

        if (!strcmp(arg, "--dump-generated-code")) {
            m_dumpGeneratedCode = true;
            continue;
        }

        if (!m_file)
            m_file = arg;
        else if (!m_entrypoint)
            m_entrypoint = arg;
        else
            printUsageStatement(false);
    }

    if (!m_file || !m_entrypoint)
        printUsageStatement(false);
}

static int runWGSL(const CommandLine& options)
{
    WGSL::Configuration configuration;


    String fileName = String::fromLatin1(options.file());
    auto handle = FileSystem::openFile(fileName, FileSystem::FileOpenMode::Read);
    if (!FileSystem::isHandleValid(handle)) {
        FileSystem::closeFile(handle);
        dataLogLn("Failed to open ", fileName);
        return EXIT_FAILURE;
    }

    auto readResult = FileSystem::readEntireFile(handle);
    FileSystem::closeFile(handle);
    auto source = emptyString();
    if (readResult.has_value())
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
        source = String::fromUTF8WithLatin1Fallback(std::span(readResult->data(), readResult->size()));
#pragma clang diagnostic pop

    auto checkResult = WGSL::staticCheck(source, std::nullopt, configuration);
    if (auto* failedCheck = std::get_if<WGSL::FailedCheck>(&checkResult)) {
        for (const auto& error : failedCheck->errors)
            dataLogLn(error);
        return EXIT_FAILURE;
    }

    auto& shaderModule = std::get<WGSL::SuccessfulCheck>(checkResult).ast;
    if (options.dumpASTAfterCheck())
        WGSL::AST::dumpAST(shaderModule);

    String entrypointName = String::fromLatin1(options.entrypoint());
    auto prepareResult = WGSL::prepare(shaderModule, entrypointName, nullptr);

    if (auto* error = std::get_if<WGSL::Error>(&prepareResult)) {
        dataLogLn(*error);
        return EXIT_FAILURE;
    }

    auto& result = std::get<WGSL::PrepareResult>(prepareResult);
    if (entrypointName != "_"_s && !result.entryPoints.contains(entrypointName)) {
        dataLogLn("WGSL source does not contain entrypoint named '", entrypointName, "'");
        return EXIT_FAILURE;
    }

    HashMap<String, WGSL::ConstantValue> constantValues;
    const auto& entryPointInformation = result.entryPoints.get(entrypointName);
    for (const auto& [originalName, constant] : entryPointInformation.specializationConstants) {
        if (!constant.defaultValue) {
            dataLogLn("Cannot use override without default value in wgslc: '", originalName, "'");
            return EXIT_FAILURE;
        }

        auto defaultValue = WGSL::evaluate(*constant.defaultValue, constantValues);
        if (!defaultValue) {
            dataLogLn("Failed to evaluate override's default value: '", originalName, "'");
            return EXIT_FAILURE;
        }

        constantValues.add(constant.mangledName, *defaultValue);
    }
    auto generationResult = WGSL::generate(shaderModule, result, constantValues);

    if (auto* error = std::get_if<WGSL::Error>(&generationResult)) {
        dataLogLn(*error);
        return EXIT_FAILURE;
    }

    auto& msl = std::get<String>(generationResult);

    if (options.dumpASTAtEnd())
        WGSL::AST::dumpAST(shaderModule);

    if (options.dumpGeneratedCode())
        printf("%s", msl.utf8().data());

    return EXIT_SUCCESS;
}

int main(int argc, char** argv)
{
    WTF::initializeMainThread();

    CommandLine commandLine(argc, argv);
    return runWGSL(commandLine);
}
