/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 24, 2025.
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
#include "DFGCodeOriginPool.h"

#if ENABLE(DFG_JIT)

namespace JSC { namespace DFG {

CallSiteIndex CodeOriginPool::addCodeOrigin(CodeOrigin codeOrigin)
{
    if (m_codeOrigins.isEmpty()
        || m_codeOrigins.last() != codeOrigin)
        m_codeOrigins.append(codeOrigin);
    unsigned index = m_codeOrigins.size() - 1;
    ASSERT(m_codeOrigins[index] == codeOrigin);
    return CallSiteIndex(index);
}

CallSiteIndex CodeOriginPool::addUniqueCallSiteIndex(CodeOrigin codeOrigin)
{
    m_codeOrigins.append(codeOrigin);
    unsigned index = m_codeOrigins.size() - 1;
    ASSERT(m_codeOrigins[index] == codeOrigin);
    return CallSiteIndex(index);
}

CallSiteIndex CodeOriginPool::lastCallSite() const
{
    RELEASE_ASSERT(m_codeOrigins.size());
    return CallSiteIndex(m_codeOrigins.size() - 1);
}

DisposableCallSiteIndex CodeOriginPool::addDisposableCallSiteIndex(CodeOrigin codeOrigin)
{
    if (!m_callSiteIndexFreeList.isEmpty()) {
        unsigned index = m_callSiteIndexFreeList.takeLast();
        m_codeOrigins[index] = codeOrigin;
        return DisposableCallSiteIndex(index);
    }

    m_codeOrigins.append(codeOrigin);
    unsigned index = m_codeOrigins.size() - 1;
    ASSERT(m_codeOrigins[index] == codeOrigin);
    return DisposableCallSiteIndex(index);
}

void CodeOriginPool::removeDisposableCallSiteIndex(DisposableCallSiteIndex callSite)
{
    RELEASE_ASSERT(callSite.bits() < m_codeOrigins.size());
    m_callSiteIndexFreeList.append(callSite.bits());
    m_codeOrigins[callSite.bits()] = CodeOrigin();
}

void CodeOriginPool::shrinkToFit()
{
    m_codeOrigins.shrinkToFit();
    m_callSiteIndexFreeList.shrinkToFit();
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
