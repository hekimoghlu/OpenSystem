/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 19, 2025.
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
#include "DFGExitProfile.h"

#if ENABLE(DFG_JIT)

#include "CodeBlock.h"
#include "VMInlines.h"

namespace JSC { namespace DFG {

void FrequentExitSite::dump(PrintStream& out) const
{
    out.print(m_bytecodeIndex, ": ", m_kind, "/", m_jitType, "/", m_inlineKind);
}

ExitProfile::ExitProfile() = default;
ExitProfile::~ExitProfile() = default;

bool ExitProfile::add(CodeBlock* owner, const FrequentExitSite& site)
{
    RELEASE_ASSERT(site.jitType() != ExitFromAnything);
    RELEASE_ASSERT(site.inlineKind() != ExitFromAnyInlineKind);

    ConcurrentJSLocker locker(owner->unlinkedCodeBlock()->m_lock);

    CODEBLOCK_LOG_EVENT(owner, "frequentExit", (site));
    
    dataLogLnIf(Options::verboseExitProfile(), pointerDump(owner), ": Adding exit site: ", site);

    ExitProfile& profile = owner->unlinkedCodeBlock()->exitProfile();
    
    // If we've never seen any frequent exits then create the list and put this site
    // into it.
    if (!profile.m_frequentExitSites) {
        profile.m_frequentExitSites = makeUnique<Vector<FrequentExitSite>>();
        profile.m_frequentExitSites->append(site);
        return true;
    }
    
    // Don't add it if it's already there. This is O(n), but that's OK, because we
    // know that the total number of places where code exits tends to not be large,
    // and this code is only used when recompilation is triggered.
    for (unsigned i = 0; i < profile.m_frequentExitSites->size(); ++i) {
        if (profile.m_frequentExitSites->at(i) == site)
            return false;
    }
    
    profile.m_frequentExitSites->append(site);
    return true;
}

Vector<FrequentExitSite> ExitProfile::exitSitesFor(BytecodeIndex bytecodeIndex)
{
    Vector<FrequentExitSite> result;
    
    if (!m_frequentExitSites)
        return result;
    
    for (unsigned i = 0; i < m_frequentExitSites->size(); ++i) {
        if (m_frequentExitSites->at(i).bytecodeIndex() == bytecodeIndex)
            result.append(m_frequentExitSites->at(i));
    }
    
    return result;
}

bool ExitProfile::hasExitSite(const ConcurrentJSLocker&, const FrequentExitSite& site) const
{
    if (!m_frequentExitSites)
        return false;
    
    for (unsigned i = m_frequentExitSites->size(); i--;) {
        if (site.subsumes(m_frequentExitSites->at(i)))
            return true;
    }
    return false;
}

QueryableExitProfile::QueryableExitProfile() = default;
QueryableExitProfile::~QueryableExitProfile() = default;

void QueryableExitProfile::initialize(UnlinkedCodeBlock* unlinkedCodeBlock)
{
    ConcurrentJSLocker locker(unlinkedCodeBlock->m_lock);
    const ExitProfile& profile = unlinkedCodeBlock->exitProfile();
    if (!profile.m_frequentExitSites)
        return;
    
    for (unsigned i = 0; i < profile.m_frequentExitSites->size(); ++i)
        m_frequentExitSites.add(profile.m_frequentExitSites->at(i));
}

} } // namespace JSC::DFG

#endif
