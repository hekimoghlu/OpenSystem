/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 5, 2022.
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
#ifndef mach_o_Version32_h
#define mach_o_Version32_h

#include <stdint.h>
#include <compare>

// common
#include "MachODefines.h"
#include "CString.h"

// mach-o
#include "Error.h"

namespace mach_o {

/*!
 * @class Version32
 *
 * @abstract
 *      Type safe wrapper for version numbers packed into 32-bits
 *      example:  X.Y[.Z] as xxxxyyzz
 */
class VIS_HIDDEN Version32
{
public:
                  constexpr Version32(uint16_t major, uint8_t minor, uint8_t micro=0) : _raw((major << 16) | (minor << 8) | micro) { }
         explicit constexpr Version32(uint32_t raw) : _raw(raw) { }
                  constexpr Version32() : _raw(0x00010000) { }

    static Error            fromString(std::string_view versString, Version32& vers,
                                       void (^ _Nullable truncationHandler)(void) = nullptr);
    const char* _Nonnull    toString(char* _Nonnull buffer) const;
    auto                    operator<=>(const Version32& other) const = default;
    uint32_t                value() const { return _raw; }
    uint32_t                major() const { return (_raw >> 16) & 0xFFFF; }
    uint32_t                minor() const { return (_raw >> 8) & 0xFF; }
private:
    uint32_t                _raw;
};

inline bool operator<(const Version32& l, const Version32& r) { return (l.value() < r.value()); }


} // namespace mach_o

#endif /* mach_o_Version32_h */
