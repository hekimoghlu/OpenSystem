/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 11, 2024.
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
#include "SIMDInfo.h"

#include <wtf/HexNumber.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace WTF {

void printInternal(PrintStream& out, JSC::SIMDLane lane)
{
    switch (lane) {
    case JSC::SIMDLane::i8x16:
        out.print("i8x16");
        break;
    case JSC::SIMDLane::i16x8:
        out.print("i16x8");
        break;
    case JSC::SIMDLane::i32x4:
        out.print("i32x4");
        break;
    case JSC::SIMDLane::f32x4:
        out.print("f32x4");
        break;
    case JSC::SIMDLane::i64x2:
        out.print("i64x2");
        break;
    case JSC::SIMDLane::f64x2:
        out.print("f64x2");
        break;
    case JSC::SIMDLane::v128:
        out.print("v128");
        break;
    default:
        RELEASE_ASSERT_NOT_REACHED();
    }
}

void printInternal(PrintStream& out, JSC::SIMDSignMode mode)
{
    switch (mode) {
    case JSC::SIMDSignMode::None:
        out.print("SignMode::None");
        break;
    case JSC::SIMDSignMode::Signed:
        out.print("SignMode::Signed");
        break;
    case JSC::SIMDSignMode::Unsigned:
        out.print("SignMode::Unsigned");
        break;
    default:
        RELEASE_ASSERT_NOT_REACHED();
    }
}

void printInternal(PrintStream& out, JSC::v128_t v)
{
    out.print("{ ", hex(v.u32x4[0], 8), ", ", hex(v.u32x4[1], 8), ", ", hex(v.u32x4[2], 8), ", ", hex(v.u32x4[3], 8), " }");
}

} // namespace WTF

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
