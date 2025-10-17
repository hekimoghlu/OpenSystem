/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 15, 2023.
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

class PutByVariant {
    WTF_MAKE_TZONE_ALLOCATED(PutByVariant);
public:
    enum Kind {
        NotSet,
        Replace,
        Transition,
        Setter,
        CustomAccessorSetter,
        Proxy,
    };
    
    PutByVariant(CacheableIdentifier identifier)
        : m_kind(NotSet)
        , m_offset(invalidOffset)
        , m_newStructure(nullptr)
        , m_identifier(WTFMove(identifier))
    {
    }
    
    PutByVariant(const PutByVariant&);
    PutByVariant& operator=(const PutByVariant&);

    static PutByVariant replace(CacheableIdentifier, const StructureSet&, PropertyOffset, bool viaGlobalProxy);
    
    static PutByVariant transition(CacheableIdentifier, const StructureSet& oldStructure, Structure* newStructure, const ObjectPropertyConditionSet&, PropertyOffset);
    
    static PutByVariant setter(CacheableIdentifier, const StructureSet&, PropertyOffset, bool viaGlobalProxy, const ObjectPropertyConditionSet&, std::unique_ptr<CallLinkStatus>);

    static PutByVariant customSetter(CacheableIdentifier, const StructureSet&, bool viaGlobalProxy, const ObjectPropertyConditionSet&, CodePtr<CustomAccessorPtrTag>, std::unique_ptr<DOMAttributeAnnotation>&&);

    static PutByVariant proxy(CacheableIdentifier, const StructureSet&, std::unique_ptr<CallLinkStatus>);
    
    Kind kind() const { return m_kind; }
    
    bool isSet() const { return kind() != NotSet; }
    bool operator!() const { return !isSet(); }
    
    const StructureSet& structure() const
    {
        ASSERT(kind() == Replace || kind() == Setter || kind() == Proxy || kind() == CustomAccessorSetter);
        return m_oldStructure;
    }
    
    const StructureSet& oldStructure() const
    {
        ASSERT(kind() == Transition || kind() == Replace || kind() == Setter || kind() == CustomAccessorSetter || kind() == Proxy);
        return m_oldStructure;
    }
    
    const StructureSet& structureSet() const
    {
        return oldStructure();
    }
    
    StructureSet& oldStructure()
    {
        ASSERT(kind() == Transition || kind() == Replace || kind() == Setter || kind() == CustomAccessorSetter || kind() == Proxy);
        return m_oldStructure;
    }
    
    StructureSet& structureSet()
    {
        return oldStructure();
    }
    
    Structure* oldStructureForTransition() const;
    
    Structure* newStructure() const
    {
        ASSERT(kind() == Transition);
        return m_newStructure;
    }
    
    void fixTransitionToReplaceIfNecessary();

    bool writesStructures() const;
    bool reallocatesStorage() const;
    bool makesCalls() const;
    
    const ObjectPropertyConditionSet& conditionSet() const { return m_conditionSet; }
    
    // We don't support intrinsics for Setters (it would be sweet if we did) but we need this for templated helpers.
    Intrinsic intrinsic() const { return NoIntrinsic; }

    // This is needed for templated helpers.
    bool isPropertyUnset() const { return false; }

    PropertyOffset offset() const
    {
        ASSERT(isSet());
        return m_offset;
    }
    
    CallLinkStatus* callLinkStatus() const
    {
        ASSERT(kind() == Setter || kind() == Proxy);
        return m_callLinkStatus.get();
    }

    bool attemptToMerge(const PutByVariant& other);
    
    DECLARE_VISIT_AGGREGATE;
    template<typename Visitor> void markIfCheap(Visitor&);
    bool finalize(VM&);
    
    void dump(PrintStream&) const;
    void dumpInContext(PrintStream&, DumpContext*) const;

    CacheableIdentifier identifier() const { return m_identifier; }

    bool overlaps(const PutByVariant& other)
    {
        if (m_viaGlobalProxy != other.m_viaGlobalProxy)
            return true;
        if (!!m_identifier != !!other.m_identifier)
            return true;
        if (m_identifier) {
            if (m_identifier != other.m_identifier)
                return false;
        }
        return structureSet().overlaps(other.structureSet());
    }

    bool viaGlobalProxy() const { return m_viaGlobalProxy; }

    CodePtr<CustomAccessorPtrTag> customAccessorSetter() const { return m_customAccessorSetter; }
    DOMAttributeAnnotation* domAttribute() const { return m_domAttribute.get(); }

private:
    bool attemptToMergeTransitionWithReplace(const PutByVariant& replace);
    
    Kind m_kind;
    bool m_viaGlobalProxy { false };
    PropertyOffset m_offset { invalidOffset };
    StructureSet m_oldStructure;
    Structure* m_newStructure { nullptr };
    ObjectPropertyConditionSet m_conditionSet;
    std::unique_ptr<CallLinkStatus> m_callLinkStatus;
    CodePtr<CustomAccessorPtrTag> m_customAccessorSetter;
    std::unique_ptr<DOMAttributeAnnotation> m_domAttribute;
    CacheableIdentifier m_identifier;
};

} // namespace JSC
