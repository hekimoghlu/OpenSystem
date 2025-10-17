/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 27, 2023.
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

// Copyright 2021 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// spirv_types.cpp:
//   Helper SPIR-V functions.

#include "spirv_types.h"

// SPIR-V tools include for AST validation.
#include <spirv-tools/libspirv.hpp>

#if defined(ANGLE_ENABLE_ASSERTS)
constexpr bool kAngleAssertEnabled = true;
#else
constexpr bool kAngleAssertEnabled = false;
#endif

namespace angle
{
namespace spirv
{

namespace
{
void ValidateSpirvMessage(spv_message_level_t level,
                          const char *source,
                          const spv_position_t &position,
                          const char *message)
{
    WARN() << "Level" << level << ": " << message;
}

spv_target_env GetEnv(const Blob &blob)
{
    switch (blob[kHeaderIndexVersion])
    {
        case kVersion_1_4:
            return SPV_ENV_VULKAN_1_1_SPIRV_1_4;
        default:
            return SPV_ENV_VULKAN_1_1;
    }
}
}  // anonymous namespace

bool Validate(const Blob &blob)
{
    if (kAngleAssertEnabled)
    {
        spvtools::SpirvTools spirvTools(GetEnv(blob));

        spvtools::ValidatorOptions options;
        options.SetFriendlyNames(false);

        spirvTools.SetMessageConsumer(ValidateSpirvMessage);
        const bool result = spirvTools.Validate(blob.data(), blob.size(), options);

        if (!result)
        {
            std::string readableSpirv;
            spirvTools.Disassemble(blob, &readableSpirv, 0);
            WARN() << "Invalid SPIR-V:\n" << readableSpirv;
        }

        return result;
    }

    // "Validate()" is only used inside an ASSERT().
    // Return false to indicate an error in case this is ever accidentally used somewhere else.
    return false;
}

void Print(const Blob &blob)
{
    spvtools::SpirvTools spirvTools(GetEnv(blob));
    std::string readableSpirv;
    spirvTools.Disassemble(blob, &readableSpirv,
                           SPV_BINARY_TO_TEXT_OPTION_COMMENT | SPV_BINARY_TO_TEXT_OPTION_INDENT |
                               SPV_BINARY_TO_TEXT_OPTION_NESTED_INDENT);
    INFO() << "Disassembled SPIR-V:\n" << readableSpirv.c_str();
}

}  // namespace spirv
}  // namespace angle
