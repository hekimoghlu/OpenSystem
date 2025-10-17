/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 1, 2022.
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

#include "AirTmpInlines.h"
#include "AirTmpWidth.h"

namespace JSC { namespace B3 { namespace Air {

inline TmpWidth::Widths& TmpWidth::widths(Tmp tmp)
{
    if (tmp.isGP()) {
        unsigned index = AbsoluteTmpMapper<GP>::absoluteIndex(tmp);
        ASSERT(index < m_widthGP.size());
        return m_widthGP[index];
    }
    unsigned index = AbsoluteTmpMapper<FP>::absoluteIndex(tmp);
    ASSERT(index < m_widthFP.size());
    return m_widthFP[index];
}

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)
