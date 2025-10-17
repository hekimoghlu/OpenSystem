/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 5, 2024.
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
#include "FTLWeight.h"

namespace JSC { namespace FTL {

class SwitchCase {
public:
    SwitchCase()
        : m_value(nullptr)
        , m_target(nullptr)
    {
    }

    SwitchCase(LValue value, LBasicBlock target, Weight weight = Weight())
        : m_value(value)
        , m_target(target)
        , m_weight(weight)
    {
    }

    LValue value() const { return m_value; }
    LBasicBlock target() const { return m_target; }
    Weight weight() const { return m_weight; }

private:
    LValue m_value;
    LBasicBlock m_target;
    Weight m_weight;
};

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)
