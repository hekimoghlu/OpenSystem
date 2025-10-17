/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 13, 2021.
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

#include <cstdint>

namespace WebCore::WebGPU {

using BufferDynamicOffset = uint32_t;
using StencilValue = uint32_t;
using SampleMask = uint32_t;
using DepthBias = int32_t;

using Size64 = uint64_t;
using IntegerCoordinate = uint32_t;
using Index32 = uint32_t;
using Size32 = uint32_t;
using SignedOffset32 = int32_t;

using FlagsConstant = uint32_t;

} // namespace WebCore::WebGPU
