/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 16, 2023.
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
#include "GetterSetterAccessCase.h"

#if ENABLE(JIT)

#include "JSCJSValueInlines.h"

namespace JSC {

GetterSetterAccessCase::GetterSetterAccessCase(VM& vm, JSCell* owner, AccessType accessType, CacheableIdentifier identifier, PropertyOffset offset, Structure* structure, const ObjectPropertyConditionSet& conditionSet, bool viaGlobalProxy, WatchpointSet* additionalSet, JSObject* customSlotBase, RefPtr<PolyProtoAccessChain>&& prototypeAccessChain)
    : Base(vm, owner, accessType, identifier, offset, structure, conditionSet, viaGlobalProxy, additionalSet, WTFMove(prototypeAccessChain))
{
    m_customSlotBase.setMayBeNull(vm, owner, customSlotBase);
}

Ref<AccessCase> GetterSetterAccessCase::create(
    VM& vm, JSCell* owner, AccessType type, CacheableIdentifier identifier, PropertyOffset offset, Structure* structure, const ObjectPropertyConditionSet& conditionSet,
    bool viaGlobalProxy, WatchpointSet* additionalSet, CodePtr<CustomAccessorPtrTag> customGetter, JSObject* customSlotBase,
    std::optional<DOMAttributeAnnotation> domAttribute, RefPtr<PolyProtoAccessChain>&& prototypeAccessChain)
{
    ASSERT(type == Getter || type == CustomValueGetter || type == CustomAccessorGetter);
    auto result = adoptRef(*new GetterSetterAccessCase(vm, owner, type, identifier, offset, structure, conditionSet, viaGlobalProxy, additionalSet, customSlotBase, WTFMove(prototypeAccessChain)));
    result->m_domAttribute = domAttribute;
    if (customGetter)
        result->m_customAccessor = customGetter;
    return result;
}

Ref<AccessCase> GetterSetterAccessCase::create(VM& vm, JSCell* owner, AccessType type, Structure* structure, CacheableIdentifier identifier, PropertyOffset offset,
    const ObjectPropertyConditionSet& conditionSet, RefPtr<PolyProtoAccessChain>&& prototypeAccessChain, bool viaGlobalProxy, 
    CodePtr<CustomAccessorPtrTag> customSetter, JSObject* customSlotBase)
{
    ASSERT(type == Setter || type == CustomValueSetter || type == CustomAccessorSetter);
    auto result = adoptRef(*new GetterSetterAccessCase(vm, owner, type, identifier, offset, structure, conditionSet, viaGlobalProxy, nullptr, customSlotBase, WTFMove(prototypeAccessChain)));
    if (customSetter)
        result->m_customAccessor = customSetter;
    return result;
}

GetterSetterAccessCase::GetterSetterAccessCase(const GetterSetterAccessCase& other)
    : Base(other)
    , m_customSlotBase(other.m_customSlotBase)
{
    m_customAccessor = other.m_customAccessor;
    m_domAttribute = other.m_domAttribute;
}

JSObject* GetterSetterAccessCase::tryGetAlternateBaseImpl() const
{
    if (auto* object = customSlotBase())
        return object;
    return Base::tryGetAlternateBaseImpl();
}

void GetterSetterAccessCase::dumpImpl(PrintStream& out, CommaPrinter& comma, Indenter& indent) const
{
    Base::dumpImpl(out, comma, indent);
    out.print(comma, "customSlotBase = ", RawPointer(customSlotBase()));
    out.print(comma, "customAccessor = ", RawPointer(m_customAccessor.taggedPtr()));
}

} // namespace JSC

#endif // ENABLE(JIT)
