/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 2, 2023.
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

#include <type_traits>

namespace JSC {

enum OpcodeSize {
    Narrow = 1,
    Wide16 = 2,
    Wide32 = 4,
};

template<OpcodeSize>
struct TypeBySize;

template<>
struct TypeBySize<OpcodeSize::Narrow> {
    using signedType = int8_t;
    using unsignedType = uint8_t;
};

template<>
struct TypeBySize<OpcodeSize::Wide16> {
    using signedType = int16_t;
    using unsignedType = uint16_t;
};

template<>
struct TypeBySize<OpcodeSize::Wide32> {
    using signedType = int32_t;
    using unsignedType = uint32_t;
};

template<OpcodeSize>
struct PaddingBySize;

template<>
struct PaddingBySize<OpcodeSize::Narrow> {
    static constexpr uint8_t value = 0;
};

template<>
struct PaddingBySize<OpcodeSize::Wide16> {
    static constexpr uint8_t value = 1;
};

template<>
struct PaddingBySize<OpcodeSize::Wide32> {
    static constexpr uint8_t value = 1;
};

template<typename Traits, OpcodeSize>
struct OpcodeIDWidthBySize;

template<typename Traits>
struct OpcodeIDWidthBySize<Traits, OpcodeSize::Narrow> {
    using opcodeType = uint8_t;
    static constexpr OpcodeSize opcodeIDSize = OpcodeSize::Narrow;
};

template<typename Traits>
struct OpcodeIDWidthBySize<Traits, OpcodeSize::Wide16> {
    using opcodeType = typename std::conditional<Traits::maxOpcodeIDWidth == OpcodeSize::Narrow, uint8_t, uint16_t>::type;
    static constexpr OpcodeSize opcodeIDSize = static_cast<OpcodeSize>(sizeof(opcodeType));
};

template<typename Traits>
struct OpcodeIDWidthBySize<Traits, OpcodeSize::Wide32> {
    using opcodeType = typename std::conditional<Traits::maxOpcodeIDWidth == OpcodeSize::Narrow, uint8_t, uint16_t>::type;
    static constexpr OpcodeSize opcodeIDSize = static_cast<OpcodeSize>(sizeof(opcodeType));
};

} // namespace JSC
