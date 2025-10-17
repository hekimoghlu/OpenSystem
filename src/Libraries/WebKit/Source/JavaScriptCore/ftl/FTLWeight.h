/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 27, 2021.
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

#include "B3FrequencyClass.h"
#include <wtf/MathExtras.h>
#include <wtf/StdLibExtras.h>

namespace JSC { namespace FTL {

class Weight {
public:
    Weight()
        : m_value(std::numeric_limits<float>::quiet_NaN())
    {
    }
    
    explicit Weight(float value)
        : m_value(value)
    {
    }
    
    bool isSet() const { return m_value == m_value; }
    bool operator!() const { return !isSet(); }
    
    float value() const { return m_value; }

    B3::FrequencyClass frequencyClass() const { return value() ? B3::FrequencyClass::Normal : B3::FrequencyClass::Rare; }
    
    // Inverse weight for a two-target branch.
    Weight inverse() const
    {
        if (!isSet())
            return Weight();
        if (value())
            return Weight(0);
        return Weight(1);
    }
    
private:
    float m_value;
};

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)
