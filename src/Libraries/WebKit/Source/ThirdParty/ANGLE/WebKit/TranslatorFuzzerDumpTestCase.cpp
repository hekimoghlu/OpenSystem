/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 27, 2024.
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
// Dumps the fuzzer sample into an ANGLE test case.
#include <iostream>
#include <fstream>
#include <vector>

#include "TranslatorFuzzerSupport.h"

// In order to link with the fuzzer that uses LLVMFuzzerMutate, provide a dummy symbol for the function.
extern "C" size_t LLVMFuzzerMutate(uint8_t*, size_t, size_t)
{
    exit(1);
    return 0;
}

static const char testHeader[] = R"cpp(
//
// Copyright 2024 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// CompilerWorks_test.cpp:
//   Some tests for shader compilation.
//

#include "GLSLANG/ShaderLang.h"
#include "angle_gl.h"
#include "compiler/translator/Compiler.h"
#include "gtest/gtest.h"

namespace sh
{
)cpp";

static const char testContent1[] = R"cpp(
{
    ShBuiltInResources resources;
    sh::InitBuiltInResources(&resources);
    ShCompileOptions options = { };
)cpp";

static const char testContent2[] = R"cpp(
    ShHandle compiler = sh::ConstructCompiler(type, spec, output, &resources);
    EXPECT_NE(static_cast<ShHandle>(0), compiler);

    const char program[] = R"TEST()cpp";

static const char testContent3[] = R"cpp()TEST";
    sh::Compile(compiler, program, 1, options);
    sh::Destruct(compiler);
}
)cpp";

static const char testFooter[] = R"cpp(
}
)cpp";

#define RETURN_STRING_IF_EQUAL(var, name) if (var == name) return #name

static std::string typeToString(uint32_t type)
{
    RETURN_STRING_IF_EQUAL(type, GL_FRAGMENT_SHADER);
    RETURN_STRING_IF_EQUAL(type, GL_VERTEX_SHADER);
    return std::string{ "static_cast<uint32_t>(" } + std::to_string(type) + ")";
}

static std::string shaderSpecToString(ShShaderSpec spec)
{
    RETURN_STRING_IF_EQUAL(spec, SH_GLES2_SPEC);
    RETURN_STRING_IF_EQUAL(spec, SH_WEBGL_SPEC);
    RETURN_STRING_IF_EQUAL(spec, SH_WEBGL2_SPEC);
    RETURN_STRING_IF_EQUAL(spec, SH_GLES3_SPEC);
    return std::string{"static_cast<ShShaderSpec>("} + std::to_string(spec) + ")";
}

static std::string shaderOutputToString(ShShaderOutput output)
{
 
    RETURN_STRING_IF_EQUAL(output, SH_MSL_METAL_OUTPUT);
    RETURN_STRING_IF_EQUAL(output, SH_ESSL_OUTPUT);
    RETURN_STRING_IF_EQUAL(output, SH_GLSL_COMPATIBILITY_OUTPUT);
    RETURN_STRING_IF_EQUAL(output, SH_GLSL_130_OUTPUT);
    RETURN_STRING_IF_EQUAL(output, SH_GLSL_140_OUTPUT);
    RETURN_STRING_IF_EQUAL(output, SH_GLSL_150_CORE_OUTPUT);
    RETURN_STRING_IF_EQUAL(output, SH_GLSL_330_CORE_OUTPUT);
    RETURN_STRING_IF_EQUAL(output, SH_GLSL_400_CORE_OUTPUT);
    RETURN_STRING_IF_EQUAL(output, SH_GLSL_410_CORE_OUTPUT);
    RETURN_STRING_IF_EQUAL(output, SH_GLSL_420_CORE_OUTPUT);
    RETURN_STRING_IF_EQUAL(output, SH_GLSL_430_CORE_OUTPUT);
    RETURN_STRING_IF_EQUAL(output, SH_GLSL_440_CORE_OUTPUT);
    RETURN_STRING_IF_EQUAL(output, SH_GLSL_450_CORE_OUTPUT);
    return std::string{"static_cast<ShShaderOutput>("} + std::to_string(output) + ")";
}

int main(int argc, const char * argv[])
{
    if (argc < 2) {
        std::cout << "usage: " << argv[0] << "testcase [testcase...]" << std::endl;
        exit(1);
    }

    std::cout << testHeader;
    for (int i = 1; i < argc; ++i) {
        std::vector<uint8_t> fileData;
        {
            std::streampos fileSize;
            {
                std::ifstream file { argv[i], std::ios::binary };
                file.seekg(0, std::ios::end);
                fileSize = file.tellg();
                file.seekg(0, std::ios::beg);
                if (fileData.size() < static_cast<size_t>(fileSize))
                    fileData.resize(static_cast<size_t>(fileSize));
                file.read(reinterpret_cast<char*>(&fileData[0]), fileSize);
            }
        }
        GLSLDumpHeader header { &fileData[0] };
        header.output = resolveShaderOutput(header.output);
        filterOptions(header.output, header.options);
        std::cout << "TEST(CompilerWorksTest, Test" << i << ")";
        std::cout << testContent1;
        const bool any = true;
        const bool msl = header.output == SH_MSL_METAL_OUTPUT;
#define COUT_EXTENSION(NAME, FORCE) if ((FORCE)) std::cout << "    resources." #NAME " = 1;" << std::endl;
        FOR_EACH_SH_BUILT_IN_RESOURCES_EXTENSION_OPTION(COUT_EXTENSION);
#undef COUT_EXTENSION
        std::cout << "    resources.MaxDualSourceDrawBuffers = 1;" << std::endl;
        std::cout << "    resources.MaxDrawBuffers = 8;" << std::endl;
#define COUT_OPTION(NAME, I, ALLOW, FORCE) if (header.options.NAME) std::cout << "    options." #NAME " = true;" << std::endl;
        FOR_EACH_SH_COMPILE_OPTIONS_BOOL_OPTION(COUT_OPTION);
#undef COUT_OPTION
        std::cout << "    uint32_t type = " << typeToString(header.type) << ";" << std::endl;
        std::cout << "    ShShaderSpec spec = " << shaderSpecToString(header.spec) << ";" << std::endl;
        std::cout << "    ShShaderOutput output = " << shaderOutputToString(header.output) << ";" << std::endl;
        std::cout << testContent2;
        std::cout << &fileData[GLSLDumpHeader::kHeaderSize];
        std::cout << testContent3;
    }
    std::cout << testFooter;
    return 0;
}
