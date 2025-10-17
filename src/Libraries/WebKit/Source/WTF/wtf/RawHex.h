/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 16, 2022.
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

namespace WTF {

// For printing integral values in hex.
// See also the printInternal function for RaswHex in PrintStream.cpp.

class RawHex {
public:
    RawHex() = default;

    explicit RawHex(int8_t value)
        : RawHex(static_cast<uint8_t>(value))
    { }
    explicit RawHex(uint8_t value)
        : RawHex(static_cast<uintptr_t>(value))
    { }

    explicit RawHex(int16_t value)
        : RawHex(static_cast<uint16_t>(value))
    { }
    explicit RawHex(uint16_t value)
        : RawHex(static_cast<uintptr_t>(value))
    { }

#if CPU(ADDRESS64) || OS(DARWIN) || OS(HAIKU)
    // These causes build errors for CPU(ADDRESS32) on some ports because int32_t
    // is already handled by intptr_t, and uint32_t is handled by uintptr_t.
    explicit RawHex(int32_t value)
        : RawHex(static_cast<uint32_t>(value))
    { }
    explicit RawHex(uint32_t value)
        : RawHex(static_cast<uintptr_t>(value))
    { }
#endif

#if CPU(ADDRESS32) || OS(DARWIN)
    // These causes build errors for CPU(ADDRESS64) on some ports because int64_t
    // is already handled by intptr_t, and uint64_t is handled by uintptr_t.
    explicit RawHex(int64_t value)
        : RawHex(static_cast<uint64_t>(value))
    { }
#if CPU(ADDRESS64) // on OS(DARWIN)
    explicit RawHex(uint64_t value)
        : RawHex(static_cast<uintptr_t>(value))
    { }
#else
    explicit RawHex(uint64_t value)
        : m_is64Bit(true)
        , m_u64(value)
    { }
#endif
#endif // CPU(ADDRESS64)

    explicit RawHex(intptr_t value)
        : RawHex(static_cast<uintptr_t>(value))
    { }
    explicit RawHex(uintptr_t value)
        : m_ptr(reinterpret_cast<void*>(value))
    { }

    const void* ptr() const { return m_ptr; }

#if !CPU(ADDRESS64)
    bool is64Bit() const { return m_is64Bit; }
    uint64_t u64() const { return m_u64; }
#endif

private:
#if !CPU(ADDRESS64)
    bool m_is64Bit { false };
#endif
    union {
        const void* m_ptr { nullptr };
#if !CPU(ADDRESS64)
        uint64_t m_u64;
#endif
    };
};

} // namespace WTF

using WTF::RawHex;
