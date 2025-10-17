/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 6, 2025.
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

#include <wtf/EnumClassOperatorOverloads.h>

namespace JSC {

enum class IterationMode : uint8_t {
    Generic = 1 << 0,
    FastArray = 1 << 1,
};

constexpr uint8_t numberOfIterationModes = 2;

OVERLOAD_BITWISE_OPERATORS_FOR_ENUM_CLASS_WITH_INTERGRALS(IterationMode);

struct IterationModeMetadata {
    uint8_t seenModes { 0 };
    static constexpr ptrdiff_t offsetOfSeenModes() { return OBJECT_OFFSETOF(IterationModeMetadata, seenModes); }
    static_assert(sizeof(decltype(seenModes)) == sizeof(IterationMode));
};

} // namespace JSC
