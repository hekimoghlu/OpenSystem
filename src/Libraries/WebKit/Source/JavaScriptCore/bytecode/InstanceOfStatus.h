/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 1, 2025.
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

#include "ConcurrentJSLock.h"
#include "ICStatusMap.h"
#include "InstanceOfVariant.h"
#include "StubInfoSummary.h"
#include <wtf/TZoneMalloc.h>

namespace JSC {

class AccessCase;
class CodeBlock;
class StructureStubInfo;

class InstanceOfStatus final {
    WTF_MAKE_TZONE_ALLOCATED(InstanceOfStatus);
public:
    enum State {
        // It's uncached so we have no information.
        NoInformation,

        // It's cached in a simple way.
        Simple,

        // It's cached for a megamorphic case.
        Megamorphic,

        // It's known to often take slow path.
        TakesSlowPath
    };

    InstanceOfStatus()
        : m_state(NoInformation)
    {
    }
    
    InstanceOfStatus(State state)
        : m_state(state)
    {
        ASSERT(state == NoInformation || state == TakesSlowPath || state == Megamorphic);
    }
    
    explicit InstanceOfStatus(StubInfoSummary summary)
    {
        switch (summary) {
        case StubInfoSummary::NoInformation:
            m_state = NoInformation;
            return;
        case StubInfoSummary::Simple:
        case StubInfoSummary::MakesCalls:
            RELEASE_ASSERT_NOT_REACHED();
            return;
        case StubInfoSummary::Megamorphic:
            m_state = Megamorphic;
            return;
        case StubInfoSummary::TakesSlowPath:
        case StubInfoSummary::TakesSlowPathAndMakesCalls:
            m_state = TakesSlowPath;
            return;
        }
        RELEASE_ASSERT_NOT_REACHED();
    }
    
    static InstanceOfStatus computeFor(CodeBlock*, ICStatusMap&, BytecodeIndex);
    
#if ENABLE(DFG_JIT)
    static InstanceOfStatus computeForStubInfo(const ConcurrentJSLocker&, VM&, StructureStubInfo*);
#endif
    
    State state() const { return m_state; }
    
    bool isSet() const { return m_state != NoInformation; }
    explicit operator bool() const { return isSet(); }
    
    bool isSimple() const { return m_state == Simple; }
    bool isMegamorphic() const { return m_state == Megamorphic; }
    bool takesSlowPath() const { return m_state == TakesSlowPath; }
    
    JSObject* commonPrototype() const;
    
    size_t numVariants() const { return m_variants.size(); }
    const Vector<InstanceOfVariant, 2>& variants() const { return m_variants; }
    const InstanceOfVariant& at(size_t index) const { return m_variants[index]; }
    const InstanceOfVariant& operator[](size_t index) const { return at(index); }

    void filter(const StructureSet&);
    
    void dump(PrintStream&) const;

private:
    void appendVariant(const InstanceOfVariant&);
    void shrinkToFit();
    
    State m_state;
    Vector<InstanceOfVariant, 2> m_variants;
};

} // namespace JSC

