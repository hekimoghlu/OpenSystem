/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 17, 2022.
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

#if ENABLE(FTL_JIT)

#include "FTLAbbreviatedTypes.h"

namespace JSC { namespace FTL {

class ValueRange {
public:
    ValueRange()
        : m_begin(nullptr)
        , m_end(nullptr)
    {
    }
    
    ValueRange(LValue begin, LValue end)
        : m_begin(begin)
        , m_end(end)
    {
    }
    
    LValue begin() const { return m_begin; }
    LValue end() const { return m_end; }
    
    void decorateInstruction(LValue loadInstruction, unsigned rangeKind) const;
    
private:
    LValue m_begin;
    LValue m_end;
};

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)
