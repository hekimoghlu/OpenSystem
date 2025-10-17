/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 3, 2023.
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

#include "AirTmp.h"
#include <wtf/IndexSet.h>

namespace JSC { namespace B3 { namespace Air {

class TmpSet {
public:
    TmpSet()
    {
    }
    
    bool add(Tmp tmp)
    {
        if (tmp.isGP())
            return m_gp.add(tmp);
        return m_fp.add(tmp);
    }
    
    bool remove(Tmp tmp)
    {
        if (tmp.isGP())
            return m_gp.remove(tmp);
        return m_fp.remove(tmp);
    }
    
    bool contains(Tmp tmp)
    {
        if (tmp.isGP())
            return m_gp.contains(tmp);
        return m_fp.contains(tmp);
    }
    
    size_t size() const
    {
        return m_gp.size() + m_fp.size();
    }
    
    bool isEmpty() const
    {
        return !size();
    }

    class iterator {
    public:
        iterator()
        {
        }
        
        iterator(BitVector::iterator gpIter, BitVector::iterator fpIter)
            : m_gpIter(gpIter)
            , m_fpIter(fpIter)
        {
        }
        
        Tmp operator*()
        {
            if (!m_gpIter.isAtEnd())
                return Tmp::tmpForAbsoluteIndex(GP, *m_gpIter);
            return Tmp::tmpForAbsoluteIndex(FP, *m_fpIter);
        }
        
        iterator& operator++()
        {
            if (!m_gpIter.isAtEnd()) {
                ++m_gpIter;
                return *this;
            }
            ++m_fpIter;
            return *this;
        }
        
        friend bool operator==(const iterator&, const iterator&) = default;
        
    private:
        BitVector::iterator m_gpIter;
        BitVector::iterator m_fpIter;
    };
    
    iterator begin() const { return iterator(m_gp.indices().begin(), m_fp.indices().begin()); }
    iterator end() const { return iterator(m_gp.indices().end(), m_fp.indices().end()); }

private:
    IndexSet<Tmp::AbsolutelyIndexed<GP>> m_gp;
    IndexSet<Tmp::AbsolutelyIndexed<FP>> m_fp;
};

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)

