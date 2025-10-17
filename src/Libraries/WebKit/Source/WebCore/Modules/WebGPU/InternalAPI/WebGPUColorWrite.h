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
#pragma once

#include "WebGPUBlendFactor.h"
#include "WebGPUBlendOperation.h"
#include "WebGPUIntegralTypes.h"
#include <cstdint>
#include <wtf/OptionSet.h>

namespace WebCore::WebGPU {

enum class ColorWrite : uint8_t {
    Red   = 1 << 0,
    Green = 1 << 1,
    Blue  = 1 << 2,
    Alpha = 1 << 3,
    All = Red | Green | Blue | Alpha
};
using ColorWriteFlags = uint32_t;
static constexpr ColorWriteFlags ColorWriteFlags_All = static_cast<ColorWriteFlags>(ColorWrite::All);

} // namespace WebCore::WebGPU
