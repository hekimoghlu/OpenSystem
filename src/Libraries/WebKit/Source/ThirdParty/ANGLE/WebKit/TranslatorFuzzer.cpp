/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 6, 2022.
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
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <random>
#include <string.h>

#include "TranslatorFuzzerSupport.h"
#include "angle_gl.h"
#include "common/string_utils.h"
#include "common/system_utils.h"
#include "compiler/translator/Compiler.h"
#include "compiler/translator/util.h"

extern "C" size_t LLVMFuzzerMutate(uint8_t *Data, size_t Size, size_t MaxSize);

using namespace sh;

namespace
{

#define ARRAY_COUNT(a) (sizeof(a) / sizeof((a)[0]))

static uint32_t allTypes[] = {
    GL_FRAGMENT_SHADER,
    GL_VERTEX_SHADER
};
static constexpr int allTypesCount = ARRAY_COUNT(allTypes);

static const struct {
    ShShaderSpec spec;
    const char* flag;
} allSpecs[] = {
    { SH_GLES2_SPEC, "i2"},
    { SH_GLES3_SPEC, "i3"},
    { SH_WEBGL_SPEC, "w" },
    { SH_WEBGL2_SPEC, "w2" },
};
static constexpr int allSpecsCount = ARRAY_COUNT(allSpecs);
static ShShaderSpec validSpecs[allSpecsCount] = {
    SH_WEBGL_SPEC,
    SH_WEBGL2_SPEC,
};
static int validSpecsCount = 2;

static const struct {
    ShShaderOutput output;
    const char* flag;
} allOutputs[] = {
    { SH_MSL_METAL_OUTPUT, "m"},
    { SH_ESSL_OUTPUT, "e" },
    { SH_GLSL_COMPATIBILITY_OUTPUT, "g" },
    { SH_GLSL_130_OUTPUT, "g130" },
    { SH_GLSL_140_OUTPUT, "g140" },
    { SH_GLSL_150_CORE_OUTPUT, "g150" },
    { SH_GLSL_330_CORE_OUTPUT, "g330" },
    { SH_GLSL_400_CORE_OUTPUT, "g400" },
    { SH_GLSL_410_CORE_OUTPUT, "g410" },
    { SH_GLSL_420_CORE_OUTPUT, "g420" },
    { SH_GLSL_430_CORE_OUTPUT, "g430" },
    { SH_GLSL_440_CORE_OUTPUT, "g440" },
    { SH_GLSL_450_CORE_OUTPUT, "g450" },
};
static constexpr int allOutputsCount = ARRAY_COUNT(allOutputs);
static ShShaderOutput validOutputs[allOutputsCount] = {
    SH_MSL_METAL_OUTPUT
};
static int validOutputsCount = 1;

static struct {
    TCompiler* translator;
    uint32_t type;
    ShShaderSpec spec;
    ShShaderOutput output;
} translators[allTypesCount * allSpecsCount * allOutputsCount];
static int translatorsCount;

void mutateOptions(ShShaderOutput output, ShCompileOptions& options)
{
    size_t mutateLength = offsetof(ShCompileOptions, metal);
    LLVMFuzzerMutate(reinterpret_cast<uint8_t*>(&options), mutateLength, mutateLength);
}

static bool initializeValidFuzzerOptions()
{
    const std::string optionsSpec = angle::GetEnvironmentVar("ANGLE_TRANSLATOR_FUZZER_OPTIONS");
    const std::vector<std::string> optionsSpecParts = angle::SplitString(optionsSpec, ";,: ", angle::KEEP_WHITESPACE, angle::SPLIT_WANT_NONEMPTY);
    const std::set<std::string> options { optionsSpecParts.begin(), optionsSpecParts.end() };
    if (options.empty())
        return true;
    int specs = 0;
    int outputs = 0;
    for (const std::string& option : options) {
        for (auto& spec : allSpecs) {
            if (option == spec.flag) {
                validSpecs[specs++] = spec.spec;
                break;
            }
        }
        for (auto& output : allOutputs) {
            if (option == output.flag) {
                validOutputs[specs++] = output.output;
                break;
            }
        }
    }
    if (specs)
        validSpecsCount = specs;
    if (outputs)
        validOutputsCount = outputs;
    return true;
}

void mutate(GLSLDumpHeader& header, unsigned seed)
{
    std::minstd_rand rnd { seed };
    switch (rnd() % 4) {
        case 0:
            header.type = allTypes[(header.type + rnd()) % allTypesCount];
            break;
        case 1:
            header.spec = validSpecs[(header.spec + rnd()) % validSpecsCount];
            break;
        case 2:
            header.output = validOutputs[(header.output + rnd()) % validOutputsCount];
            break;
        case 3:
            mutateOptions(header.output, header.options);
            break;
    }
}

bool initializeTranslators()
{
    if (!sh::Initialize())
        return false;
    for (int typeIndex = 0; typeIndex < allTypesCount; ++typeIndex) {
        auto type = allTypes[typeIndex];
        for (int specIndex = 0; specIndex < validSpecsCount; ++specIndex) {
            auto spec = validSpecs[specIndex];
            for (int outputIndex = 0; outputIndex < validOutputsCount; ++outputIndex) {
                auto output = validOutputs[outputIndex];
                TCompiler* translator = ConstructCompiler(type, spec, output);
                ShBuiltInResources resources;
                sh::InitBuiltInResources(&resources);
                const bool any = true;
                const bool msl = output == SH_MSL_METAL_OUTPUT;
#define ENABLE_EXTENSION(name, forced) if ((forced)) resources.name = 1;
                FOR_EACH_SH_BUILT_IN_RESOURCES_EXTENSION_OPTION(ENABLE_EXTENSION);
#undef ENABLE_EXTENSION
                resources.MaxDualSourceDrawBuffers = 1;
                resources.MaxDrawBuffers = 8;
                if (!translator->Init(resources))
                    return false;
                translators[translatorsCount++] = { translator, type, spec, output };
            }
        }
    }
    return true;
}

TCompiler* getTranslator(uint32_t type, ShShaderSpec spec, ShShaderOutput output)
{
    for (int i = 0; i < translatorsCount; ++i) {
        if (translators[i].type == type && translators[i].spec == spec && translators[i].output == output)
            return translators[i].translator;
    }
    return nullptr;
}

void initializeFuzzer()
{
    static int i = [] {
        if (!initializeValidFuzzerOptions() || !initializeTranslators())
            exit(1);
        return 0;
    }();
    ANGLE_UNUSED_VARIABLE(i);
}

}

void filterOptions(ShShaderOutput output, ShCompileOptions& options)
{
    const bool any = true;
    const bool none = false;
    const bool msl = output == SH_MSL_METAL_OUTPUT;
    const bool glsl = IsOutputGLSL(output) || IsOutputESSL(output);
#if defined(ANGLE_PLATFORM_APPLE)
    const bool appleGLSL = glsl;
#else
    const bool appleGLSL = false;
#endif
    const bool hlsl = IsOutputHLSL(output);
    const bool spirvVk = output == SH_SPIRV_VULKAN_OUTPUT;
    const bool wgsl = output == SH_WGSL_OUTPUT;
#define CHECK_VALID_OPTION(name, i, allowed, forced) options.name = (allowed) ? options.name : (forced);
    FOR_EACH_SH_COMPILE_OPTIONS_BOOL_OPTION(CHECK_VALID_OPTION)
#undef CHECK_VALID_OPTION
}

ShShaderOutput resolveShaderOutput(ShShaderOutput output)
{
    // Constants in ShaderLang.h version 363.
    switch (static_cast<unsigned>(output)) {
    case 0x8B45:
        return SH_ESSL_OUTPUT;
    case 0x8B46:
        return SH_GLSL_COMPATIBILITY_OUTPUT;
    case 0x8B47:
        return SH_GLSL_130_OUTPUT;
    case 0x8B80:
        return SH_GLSL_140_OUTPUT;
    case 0x8B81:
        return SH_GLSL_150_CORE_OUTPUT;
    case 0x8B82:
        return SH_GLSL_330_CORE_OUTPUT;
    case 0x8B83:
        return SH_GLSL_400_CORE_OUTPUT;
    case 0x8B84:
        return SH_GLSL_410_CORE_OUTPUT;
    case 0x8B85:
        return SH_GLSL_420_CORE_OUTPUT;
    case 0x8B86:
        return SH_GLSL_430_CORE_OUTPUT;
    case 0x8B87:
        return SH_GLSL_440_CORE_OUTPUT;
    case 0x8B88:
        return SH_GLSL_450_CORE_OUTPUT;
    case 0x8B48:
        return SH_HLSL_3_0_OUTPUT;
    case 0x8B49:
        return SH_HLSL_4_1_OUTPUT;
    case 0x8B4B:
        return SH_SPIRV_VULKAN_OUTPUT;
    case 0x8B4D:
        return SH_MSL_METAL_OUTPUT;
    case 0x8B4E:
        return SH_WGSL_OUTPUT;
    };
    return output;
}

extern "C" size_t LLVMFuzzerCustomMutator(uint8_t *data, size_t size, size_t maxSize, unsigned int seed)
{
    initializeFuzzer();
    if (maxSize < GLSLDumpHeader::kHeaderSize + 2)
        return size;
    if (size < GLSLDumpHeader::kHeaderSize + 2)
        size = GLSLDumpHeader::kHeaderSize + 2;
    GLSLDumpHeader header { data };
    if (std::minstd_rand { seed }() % 100 == 0) {
        mutate(header, seed);
        header.write(data);
    }
    data += GLSLDumpHeader::kHeaderSize;
    size -= GLSLDumpHeader::kHeaderSize;
    maxSize -= GLSLDumpHeader::kHeaderSize;
    size_t newSize = LLVMFuzzerMutate(data, size - 1, maxSize - 1);
    if (data[newSize] != '\0')
        data[newSize++] = '\0';
    return GLSLDumpHeader::kHeaderSize + newSize;
}

extern "C" int LLVMFuzzerTestOneInput (const uint8_t* data, size_t size)
{
    initializeFuzzer();
    if (size < GLSLDumpHeader::kHeaderSize + 2)
        return 0;
    // Make sure the rest of data will be a valid C string so that we don't have to copy it.
    if (data[size - 1] != 0)
        return 0;

    GLSLDumpHeader header { data };
    header.output = resolveShaderOutput(header.output);
    filterOptions(header.output, header.options);
    auto* translator = getTranslator(header.type, header.spec, header.output);
    if (!translator)
        return 0;
    size -= GLSLDumpHeader::kHeaderSize;
    data += GLSLDumpHeader::kHeaderSize;
    const char *shaderStrings[] = { reinterpret_cast<const char *>(data) };
    translator->compile(shaderStrings, 1, header.options);
    return 0;
}
