/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 27, 2022.
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
#include "MetalCodeGenerator.h"

#include "AST.h"
#include "MetalFunctionWriter.h"
#include <wtf/DataLog.h>
#include <wtf/text/StringBuilder.h>

#if PLATFORM(COCOA)
#include <notify.h>
#endif
namespace WGSL {

namespace Metal {

static StringView metalCodePrologue()
{
    return StringView {
        "#include <metal_stdlib>\n"
        "#include <metal_types>\n"
        "\n"
        "using namespace metal;\n"
        "\n"_s
    };

}

#if PLATFORM(COCOA)
static void dumpMetalCodeIfNeeded(StringBuilder& stringBuilder)
{
    static bool dumpMetalCode = false;
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        int dumpMetalCodeToken;
        notify_register_dispatch("com.apple.WebKit.WebGPU.TogglePrintMetalCode", &dumpMetalCodeToken, dispatch_get_main_queue(), ^(int) {
            dumpMetalCode = !dumpMetalCode;
        });
    });

    if (dumpMetalCode) {
        dataLogLn("Generated Metal code:");
        dataLogLn(stringBuilder.toString());
    }
}
#endif

String generateMetalCode(ShaderModule& shaderModule, PrepareResult& prepareResult, const HashMap<String, ConstantValue>& constantValues)
{
    StringBuilder stringBuilder;
    stringBuilder.append(metalCodePrologue());

    Metal::emitMetalFunctions(stringBuilder, shaderModule, prepareResult, constantValues);

#if PLATFORM(COCOA)
    dumpMetalCodeIfNeeded(stringBuilder);
#endif

    return stringBuilder.toString();
}

} // namespace Metal
} // namespace WGSL
