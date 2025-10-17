/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 5, 2025.
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

#include "AirCode.h"
#include "AirTmp.h"
#include <wtf/IndexMap.h>

namespace JSC { namespace B3 { namespace Air {

// As an alternative to this, you could use IndexMap<Tmp::LinearlyIndexed, ...>, but this would fail
// as soon as you added a new GP tmp.

template<typename Value>
class TmpMap {
public:
    TmpMap() = default;
    TmpMap(TmpMap&&) = default;
    TmpMap& operator=(TmpMap&&) = default;
    
    template<typename... Args>
    TmpMap(Code& code, const Args&... args)
        : m_gp(Tmp::absoluteIndexEnd(code, GP), args...)
        , m_fp(Tmp::absoluteIndexEnd(code, FP), args...)
    {
    }
    
    template<typename... Args>
    void resize(Code& code, const Args&... args)
    {
        m_gp.resize(Tmp::absoluteIndexEnd(code, GP), args...);
        m_fp.resize(Tmp::absoluteIndexEnd(code, FP), args...);
    }
    
    template<typename... Args>
    void clear(const Args&... args)
    {
        m_gp.remove(args...);
        m_fp.remove(args...);
    }
    
    const Value& operator[](Tmp tmp) const
    {
        if (tmp.isGP())
            return m_gp[tmp];
        return m_fp[tmp];
    }

    Value& operator[](Tmp tmp)
    {
        if (tmp.isGP())
            return m_gp[tmp];
        return m_fp[tmp];
    }
    
    template<typename PassedValue>
    void append(Tmp tmp, PassedValue&& value)
    {
        if (tmp.isGP())
            m_gp.append(tmp, std::forward<PassedValue>(value));
        else
            m_fp.append(tmp, std::forward<PassedValue>(value));
    }

private:
    IndexMap<Tmp::AbsolutelyIndexed<GP>, Value> m_gp;
    IndexMap<Tmp::AbsolutelyIndexed<FP>, Value> m_fp;
};

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)

