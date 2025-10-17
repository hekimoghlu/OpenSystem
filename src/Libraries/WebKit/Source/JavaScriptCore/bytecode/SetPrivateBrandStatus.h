/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 31, 2022.
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

#include "CacheableIdentifier.h"
#include "CodeOrigin.h"
#include "ConcurrentJSLock.h"
#include "ExitFlag.h"
#include "ICStatusMap.h"
#include "SetPrivateBrandVariant.h"
#include "StubInfoSummary.h"
#include <wtf/TZoneMalloc.h>

namespace JSC {

class AccessCase;
class CodeBlock;
class StructureSet;
class StructureStubInfo;

class SetPrivateBrandStatus final {
    WTF_MAKE_TZONE_ALLOCATED(SetPrivateBrandStatus);
public:
    enum State : uint8_t {
        // It's uncached so we have no information.
        NoInformation,
        // It's cached for a simple access.
        Simple,
        // It will likely take the slow path.
        LikelyTakesSlowPath,
        // It has been seen to take the slow path.
        ObservedTakesSlowPath,
    };

    SetPrivateBrandStatus()
        : m_state(NoInformation)
    {
    }

    explicit SetPrivateBrandStatus(State state)
        : m_state(state)
    {
        ASSERT(state != Simple);
    }

    static SetPrivateBrandStatus computeFor(CodeBlock* baselineBlock, ICStatusMap& baselineMap, ICStatusContextStack& dfgContextStack, CodeOrigin);

    State state() const { return m_state; }

    bool isSet() const { return m_state != NoInformation; }
    bool operator!() const { return !isSet(); }
    bool observedSlowPath() const { return m_state == ObservedTakesSlowPath; }
    bool isSimple() const { return m_state == Simple; }
    const Vector<SetPrivateBrandVariant, 1>& variants() { return m_variants; }
    CacheableIdentifier singleIdentifier() const;

    SetPrivateBrandStatus slowVersion() const;

    // Attempts to reduce the set of variants to fit the given structure set. This may be approximate.
    void filter(const StructureSet&);

    DECLARE_VISIT_AGGREGATE;
    template<typename Visitor> void markIfCheap(Visitor&);
    bool finalize(VM&);

    bool appendVariant(const SetPrivateBrandVariant&);
    void shrinkToFit();

    void dump(PrintStream&) const;

private:
    explicit SetPrivateBrandStatus(StubInfoSummary, StructureStubInfo&);
    void merge(const SetPrivateBrandStatus&);

    static SetPrivateBrandStatus computeForBaseline(CodeBlock*, ICStatusMap&, BytecodeIndex, ExitFlag);
#if ENABLE(JIT)
    static SetPrivateBrandStatus computeForStubInfoWithoutExitSiteFeedback(
        const ConcurrentJSLocker&, CodeBlock* profiledBlock, StructureStubInfo*);
#endif

    Vector<SetPrivateBrandVariant, 1> m_variants;
    State m_state;
};

} // namespace JSC
