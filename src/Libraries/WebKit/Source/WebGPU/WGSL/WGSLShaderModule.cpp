/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 2, 2025.
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
#include "WGSLShaderModule.h"

#include "WGSL.h"
#include <wtf/TZoneMallocInlines.h>

namespace WGSL {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ShaderModule);

std::optional<Error> ShaderModule::validateOverrides(const HashMap<String, ConstantValue>& constantValues)
{
    for (const auto& [expression, validators] : m_overrideValidations) {
        auto maybeValue = evaluate(*expression, constantValues);
        if (!maybeValue)
            return { Error("failed to evaluate override expression"_s, expression->span()) };

        for (const auto& validator : validators) {
            if (auto maybeError = validator(*maybeValue))
                return { Error(*maybeError, expression->span()) };
        }

    }
    return std::nullopt;
}

} // namespace WGSL
