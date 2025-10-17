/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 16, 2023.
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

#if ENABLE(B3_JIT)

#include "AirArg.h"

namespace JSC { namespace B3 { namespace Air {

static constexpr uint8_t formRoleShift = 0;
static constexpr uint8_t formRoleMask = 0b1111;
static constexpr uint8_t formBankShift = 4;
static constexpr uint8_t formBankMask = 0b01;
static constexpr uint8_t formWidthShift = 5;
static constexpr uint8_t formWidthMask = 0b111;

inline constexpr uint8_t encodeFormWidth(Width width)
{
    switch (width) {
    case Width8:
        return 0b001; // 0 is invalid
    case Width16:
        return 0b010;
    case Width32:
        return 0b011;
    case Width64:
        return 0b100;
    case Width128:
        return 0b101;
    }
}

#define ENCODE_INST_FORM(role, bank, width) (static_cast<uint8_t>(role) << formRoleShift | static_cast<uint8_t>(bank) << formBankShift | encodeFormWidth(width) << formWidthShift)

#define INVALID_INST_FORM (0)

JS_EXPORT_PRIVATE extern const uint8_t g_formTable[];

inline Arg::Role decodeFormRole(uint8_t value)
{
    return static_cast<Arg::Role>((value >> formRoleShift) & formRoleMask);
}

inline Bank decodeFormBank(uint8_t value)
{
    return static_cast<Bank>((value >> formBankShift) & formBankMask);
}

inline Width decodeFormWidth(uint8_t value)
{
    switch ((value >> formWidthShift) & formWidthMask) {
    case 0b001:
        return Width8;
    case 0b010:
        return Width16;
    case 0b011:
        return Width32;
    case 0b100:
        return Width64;
    case 0b101:
        return Width128;
    default:
        RELEASE_ASSERT_NOT_REACHED();
        return Width64;
    }
}

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)

