/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 8, 2025.
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
#include "CallLinkStatus.h"
#include "ObjectPropertyConditionSet.h"
#include "PropertyOffset.h"
#include "StructureSet.h"
#include <wtf/TZoneMalloc.h>

namespace JSC {
namespace DOMJIT {
class GetterSetter;
}

class CallLinkStatus;
class InByStatus;
struct DumpContext;

class InByVariant {
    WTF_MAKE_TZONE_ALLOCATED(InByVariant);
public:
    InByVariant(CacheableIdentifier, const StructureSet& = StructureSet(), PropertyOffset = invalidOffset, const ObjectPropertyConditionSet& = ObjectPropertyConditionSet(), std::unique_ptr<CallLinkStatus> = nullptr);
    ~InByVariant();

    InByVariant(const InByVariant&);
    InByVariant& operator=(const InByVariant&);

    bool isSet() const { return !!m_structureSet.size(); }
    explicit operator bool() const { return isSet(); }
    const StructureSet& structureSet() const { return m_structureSet; }
    StructureSet& structureSet() { return m_structureSet; }

    // A non-empty condition set means that this is a prototype in-hit/in-miss.
    const ObjectPropertyConditionSet& conditionSet() const { return m_conditionSet; }

    PropertyOffset offset() const { return m_offset; }
    CallLinkStatus* callLinkStatus() const { return m_callLinkStatus.get(); }

    bool isHit() const { return offset() != invalidOffset; }

    bool attemptToMerge(const InByVariant& other);
    
    DECLARE_VISIT_AGGREGATE;
    template<typename Visitor> void markIfCheap(Visitor&);
    bool finalize(VM&);

    void dump(PrintStream&) const;
    void dumpInContext(PrintStream&, DumpContext*) const;

    CacheableIdentifier identifier() const { return m_identifier; }

    bool overlaps(const InByVariant& other)
    {
        if (!!m_identifier != !!other.m_identifier)
            return true;
        if (m_identifier) {
            if (m_identifier != other.m_identifier)
                return false;
        }
        return structureSet().overlaps(other.structureSet());
    }

private:
    friend class InByStatus;

    StructureSet m_structureSet;
    ObjectPropertyConditionSet m_conditionSet;
    PropertyOffset m_offset;
    CacheableIdentifier m_identifier;
    std::unique_ptr<CallLinkStatus> m_callLinkStatus;
};

} // namespace JSC
