/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 4, 2022.
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

#if ENABLE(DFG_JIT)

#include <wtf/PrintStream.h>

namespace JSC { namespace DFG {

// Utility class for epoch-based analyses.

class Epoch {
public:
    Epoch()
        : m_epoch(s_none)
    {
    }
    
    static Epoch fromUnsigned(unsigned value)
    {
        Epoch result;
        result.m_epoch = value;
        return result;
    }
    
    unsigned toUnsigned() const
    {
        return m_epoch;
    }
    
    static Epoch first()
    {
        Epoch result;
        result.m_epoch = s_first;
        return result;
    }
    
    bool operator!() const
    {
        return m_epoch == s_none;
    }
    
    explicit operator bool() const
    {
        return !!*this;
    }
    
    Epoch next() const
    {
        Epoch result;
        result.m_epoch = m_epoch + 1;
        return result;
    }
    
    void bump()
    {
        *this = next();
    }
    
    friend bool operator==(const Epoch&, const Epoch&) = default;
    
    bool operator<(const Epoch& other) const
    {
        return m_epoch < other.m_epoch;
    }
    
    bool operator>(const Epoch& other) const
    {
        return other < *this;
    }
    
    bool operator<=(const Epoch& other) const
    {
        return !(*this > other);
    }
    
    bool operator>=(const Epoch& other) const
    {
        return !(*this < other);
    }
    
    void dump(PrintStream&) const;
    
private:
    static constexpr unsigned s_none = 0;
    static constexpr unsigned s_first = 1;
    
    unsigned m_epoch;
};

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
