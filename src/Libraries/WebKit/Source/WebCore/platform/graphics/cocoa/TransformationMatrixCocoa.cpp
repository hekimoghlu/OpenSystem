/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 6, 2023.
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
#include "TransformationMatrix.h"

#if PLATFORM(COCOA)

#include <simd/simd.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace WebCore {

TransformationMatrix::TransformationMatrix(const simd_float4x4& t)
: TransformationMatrix(t.columns[0][0], t.columns[0][1], t.columns[0][2], t.columns[0][3],
    t.columns[1][0], t.columns[1][1], t.columns[1][2], t.columns[1][3],
    t.columns[2][0], t.columns[2][1], t.columns[2][2], t.columns[2][3],
    t.columns[3][0], t.columns[3][1], t.columns[3][2], t.columns[3][3])
{
}

TransformationMatrix::operator simd_float4x4() const
{
    return simd_float4x4 {
        simd_float4 { (float)m11(), (float)m12(), (float)m13(), (float)m14() },
        simd_float4 { (float)m21(), (float)m22(), (float)m23(), (float)m24() },
        simd_float4 { (float)m31(), (float)m32(), (float)m33(), (float)m34() },
        simd_float4 { (float)m41(), (float)m42(), (float)m43(), (float)m44() }
    };
}

}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // PLATFORM(COCOA)
