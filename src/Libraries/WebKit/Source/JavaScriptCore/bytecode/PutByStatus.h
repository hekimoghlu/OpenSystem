/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 11, 2023.
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
#include "ExitFlag.h"
#include "ICStatusMap.h"
#include "PrivateFieldPutKind.h"
#include "PutByVariant.h"
#include "StubInfoSummary.h"

namespace JSC {

class CodeBlock;
class VM;
class JSGlobalObject;
class Structure;
class StructureChain;
class StructureStubInfo;

typedef UncheckedKeyHashMap<CodeOrigin, StructureStubInfo*, CodeOriginApproximateHash> StubInfoMap;

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(PutByStatus);

class PutByStatus final {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(PutByStatus);
public:
    enum State {
        // It's uncached so we have no information.
        NoInformation,
        // It's cached as a simple store of some kind.
        Simple,
        // It's cached for a custom accessor with a possible structure chain.
        CustomAccessor,
        // It's cached for a proxy object.
        ProxyObject,
        // It's cached for a megamorphic case.
        Megamorphic,
        // It will likely take the slow path.
        LikelyTakesSlowPath,
        // It's known to take slow path. We also observed that the slow path was taken on StructureStubInfo.
        ObservedTakesSlowPath,
        // It will likely take the slow path and will make calls.
        MakesCalls,
        // It known to take paths that make calls. We also observed that the slow path was taken on StructureStubInfo.
        ObservedSlowPathAndMakesCalls,
    };
    
    PutByStatus()
        : m_state(NoInformation)
    {
    }
    
    explicit PutByStatus(State state)
        : m_state(state)
    {
#if ASSERT_ENABLED
        switch (m_state) {
        case NoInformation:
        case LikelyTakesSlowPath:
        case ObservedTakesSlowPath:
        case MakesCalls:
        case ObservedSlowPathAndMakesCalls:
        case Megamorphic:
        case ProxyObject:
            break;
        default:
            RELEASE_ASSERT_NOT_REACHED();
            break;
        }
#endif
    }
    
    explicit PutByStatus(StubInfoSummary, StructureStubInfo&);
    
    PutByStatus(const PutByVariant& variant)
        : m_state(Simple)
    {
        m_variants.append(variant);
    }
    
    static PutByStatus computeFor(CodeBlock*, ICStatusMap&, BytecodeIndex, ExitFlag, CallLinkStatus::ExitSiteData);
    static PutByStatus computeFor(JSGlobalObject*, const StructureSet&, CacheableIdentifier, bool isDirect, PrivateFieldPutKind);
    
    static PutByStatus computeFor(CodeBlock* baselineBlock, ICStatusMap& baselineMap, ICStatusContextStack&, CodeOrigin);

#if ENABLE(JIT)
    static PutByStatus computeForStubInfo(const ConcurrentJSLocker&, CodeBlock* baselineBlock, StructureStubInfo*, CodeOrigin);
#endif
    
    State state() const { return m_state; }
    
    bool isSet() const { return m_state != NoInformation; }
    bool operator!() const { return m_state == NoInformation; }
    bool isSimple() const { return m_state == Simple; }
    bool isCustomAccessor() const { return m_state == CustomAccessor; }
    bool isMegamorphic() const { return m_state == Megamorphic; }
    bool isProxyObject() const { return m_state == ProxyObject; }
    bool takesSlowPath() const
    {
        switch (m_state) {
        case CustomAccessor:
        case Megamorphic:
        case LikelyTakesSlowPath:
        case ObservedTakesSlowPath:
            return true;
        default:
            return false;
        }
    }
    bool makesCalls() const;
    PutByStatus slowVersion() const;
    bool observedStructureStubInfoSlowPath() const { return m_state == ObservedTakesSlowPath || m_state == ObservedSlowPathAndMakesCalls; }
    
    size_t numVariants() const { return m_variants.size(); }
    const Vector<PutByVariant, 1>& variants() const { return m_variants; }
    const PutByVariant& at(size_t index) const { return m_variants[index]; }
    const PutByVariant& operator[](size_t index) const { return at(index); }
    CacheableIdentifier singleIdentifier() const;

    bool viaGlobalProxy() const
    {
        if (m_variants.isEmpty())
            return false;
        return m_variants.first().viaGlobalProxy();
    }

    DECLARE_VISIT_AGGREGATE;
    template<typename Visitor> void markIfCheap(Visitor&);
    bool finalize(VM&);
    
    void merge(const PutByStatus&);
    
    void filter(const StructureSet&);
    
    void dump(PrintStream&) const;
    
private:
#if ENABLE(JIT)
    static PutByStatus computeForStubInfo(const ConcurrentJSLocker&, CodeBlock*, StructureStubInfo*, CallLinkStatus::ExitSiteData, CodeOrigin);
#endif
    static PutByStatus computeFromLLInt(CodeBlock*, BytecodeIndex);
    
    bool appendVariant(const PutByVariant&);
    void shrinkToFit();
    
    State m_state;
    Vector<PutByVariant, 1> m_variants;
};

} // namespace JSC
