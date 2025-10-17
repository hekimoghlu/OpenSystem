/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 26, 2024.
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

#include "CallLinkStatus.h"
#include "CodeOrigin.h"
#include "ConcurrentJSLock.h"
#include "ICStatusMap.h"
#include "InByVariant.h"
#include "StubInfoSummary.h"
#include <wtf/TZoneMalloc.h>

namespace JSC {

class AccessCase;
class CodeBlock;
class StructureStubInfo;

class InByStatus final {
    WTF_MAKE_TZONE_ALLOCATED(InByStatus);
public:
    enum State {
        // It's uncached so we have no information.
        NoInformation,
        // It's cached for a simple access to a known object property with
        // a possible structure chain and a possible specific value.
        Simple,
        // It's cached for a proxy object case.
        ProxyObject,
        // It's cached for a megamorphic case.
        Megamorphic,
        // It's known to often take slow path.
        TakesSlowPath,
    };

    InByStatus() = default;

    InByStatus(State state)
        : m_state(state)
    {
        ASSERT(state != Simple);
    }

    explicit InByStatus(StubInfoSummary summary)
    {
        switch (summary) {
        case StubInfoSummary::NoInformation:
            m_state = NoInformation;
            return;
        case StubInfoSummary::Simple:
        case StubInfoSummary::Megamorphic:
        case StubInfoSummary::MakesCalls:
            RELEASE_ASSERT_NOT_REACHED();
            return;
        case StubInfoSummary::TakesSlowPath:
        case StubInfoSummary::TakesSlowPathAndMakesCalls:
            m_state = TakesSlowPath;
            return;
        }
        RELEASE_ASSERT_NOT_REACHED();
    }

    static InByStatus computeFor(CodeBlock*, ICStatusMap&, BytecodeIndex, ExitFlag, CallLinkStatus::ExitSiteData callExitSiteData, CodeOrigin);
    static InByStatus computeFor(CodeBlock* baselineBlock, ICStatusMap& baselineMap, ICStatusContextStack&, CodeOrigin);

    State state() const { return m_state; }

    bool isSet() const { return m_state != NoInformation; }
    explicit operator bool() const { return isSet(); }
    bool isSimple() const { return m_state == Simple; }
    bool isMegamorphic() const { return m_state == Megamorphic; }
    bool isProxyObject() const { return m_state == ProxyObject; }

    size_t numVariants() const { return m_variants.size(); }
    const Vector<InByVariant, 1>& variants() const { return m_variants; }
    const InByVariant& at(size_t index) const { return m_variants[index]; }
    const InByVariant& operator[](size_t index) const { return at(index); }

    bool takesSlowPath() const { return m_state == Megamorphic || m_state == TakesSlowPath; }
    
    void merge(const InByStatus&);

    // Attempts to reduce the set of variants to fit the given structure set. This may be approximate.
    void filter(const StructureSet&);
    
    DECLARE_VISIT_AGGREGATE;
    template<typename Visitor> void markIfCheap(Visitor&);
    bool finalize(VM&);

    void dump(PrintStream&) const;

    CacheableIdentifier singleIdentifier() const;

private:
#if ENABLE(DFG_JIT)
    static InByStatus computeForStubInfoWithoutExitSiteFeedback(const ConcurrentJSLocker&, CodeBlock*, StructureStubInfo*, CallLinkStatus::ExitSiteData, CodeOrigin);
#endif
    bool appendVariant(const InByVariant&);
    void shrinkToFit();

    State m_state { NoInformation };
    Vector<InByVariant, 1> m_variants;
};

} // namespace JSC
