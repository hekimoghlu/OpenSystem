/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 20, 2025.
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

extern "C" {
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t);
int LLVMFuzzerInitialize(int *argc, char ***argv);
} // extern "C"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    WTF::initializeMainThread();

    WGSL::Configuration configuration;
    auto source = String::fromUTF8WithLatin1Fallback({ data, size });
    auto checkResult = WGSL::staticCheck(source, std::nullopt, configuration);
    if (auto* successfulCheck = std::get_if<WGSL::SuccessfulCheck>(&checkResult)) {
        auto& shaderModule = successfulCheck->ast;
        WGSL::prepare(shaderModule, "main"_str, nullptr);
    }

    return 0;
}
