/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 25, 2022.
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
#include <wtf/Box.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {
namespace DOMJIT {
class GetterSetter;
}

class CallLinkStatus;
class GetByStatus;
struct DumpContext;

class GetByVariant {
    WTF_MAKE_TZONE_ALLOCATED(GetByVariant);
public:
    GetByVariant(
        CacheableIdentifier,
        const StructureSet& = StructureSet(), bool viaGlobalProxy = false, PropertyOffset = invalidOffset,
        const ObjectPropertyConditionSet& = ObjectPropertyConditionSet(),
        std::unique_ptr<CallLinkStatus> = nullptr,
        JSFunction* = nullptr,
        CodePtr<CustomAccessorPtrTag> customAccessorGetter = nullptr,
        std::unique_ptr<DOMAttributeAnnotation> = nullptr);

    ~GetByVariant();
    
    GetByVariant(const GetByVariant&);
    GetByVariant& operator=(const GetByVariant&);
    
    bool isSet() const { return !!m_structureSet.size(); }
    explicit operator bool() const { return isSet(); }
    const StructureSet& structureSet() const { return m_structureSet; }
    StructureSet& structureSet() { return m_structureSet; }

    // A non-empty condition set means that this is a prototype load.
    const ObjectPropertyConditionSet& conditionSet() const { return m_conditionSet; }
    
    PropertyOffset offset() const { return m_offset; }
    CallLinkStatus* callLinkStatus() const { return m_callLinkStatus.get(); }
    JSFunction* intrinsicFunction() const { return m_intrinsicFunction; }
    Intrinsic intrinsic() const { return m_intrinsicFunction ? m_intrinsicFunction->intrinsic() : NoIntrinsic; }
    CodePtr<CustomAccessorPtrTag> customAccessorGetter() const { return m_customAccessorGetter; }
    DOMAttributeAnnotation* domAttribute() const { return m_domAttribute.get(); }

    bool isPropertyUnset() const { return offset() == invalidOffset; }

    bool attemptToMerge(const GetByVariant& other);
    
    DECLARE_VISIT_AGGREGATE;
    template<typename Visitor> void markIfCheap(Visitor&);
    bool finalize(VM&);
    
    void dump(PrintStream&) const;
    void dumpInContext(PrintStream&, DumpContext*) const;

    CacheableIdentifier identifier() const { return m_identifier; }

    bool overlaps(const GetByVariant& other)
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

private:
    friend class GetByStatus;

    bool canMergeIntrinsicStructures(const GetByVariant&) const;
    
    StructureSet m_structureSet;
    ObjectPropertyConditionSet m_conditionSet;
    bool m_viaGlobalProxy { false };
    PropertyOffset m_offset;
    std::unique_ptr<CallLinkStatus> m_callLinkStatus;
    JSFunction* m_intrinsicFunction;
    CodePtr<CustomAccessorPtrTag> m_customAccessorGetter;
    std::unique_ptr<DOMAttributeAnnotation> m_domAttribute;
    CacheableIdentifier m_identifier;
};

} // namespace JSC
