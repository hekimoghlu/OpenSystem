/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 10, 2023.
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

class WeightedTarget {
public:
    WeightedTarget()
        : m_target(nullptr)
    {
    }
    
    WeightedTarget(LBasicBlock target, Weight weight)
        : m_target(target)
        , m_weight(weight)
    {
    }
    
    WeightedTarget(LBasicBlock target, float weight)
        : m_target(target)
        , m_weight(weight)
    {
    }
    
    LBasicBlock target() const { return m_target; }
    Weight weight() const { return m_weight; }
    
    B3::FrequentedBlock frequentedBlock() const
    {
        return B3::FrequentedBlock(target(), weight().frequencyClass());
    }
    
private:
    LBasicBlock m_target;
    Weight m_weight;
};

// Helpers for creating weighted targets for statically known (or unknown) branch
// profiles.

inline WeightedTarget usually(LBasicBlock block)
{
    return WeightedTarget(block, 1);
}

inline WeightedTarget rarely(LBasicBlock block)
{
    return WeightedTarget(block, 0);
}

// Currently in B3 this is the equivalent of "usually", but we like to make the distinction in
// case we ever make B3 support proper branch weights. We used to do that in LLVM.
inline WeightedTarget unsure(LBasicBlock block)
{
    return WeightedTarget(block, Weight());
}

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)
