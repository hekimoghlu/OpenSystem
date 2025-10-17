/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 6, 2024.
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

#if ENABLE(JIT)

#include "MacroAssemblerCodeRef.h"
#include "ProxyableAccessCase.h"

namespace JSC {

class GetterSetterAccessCase final : public ProxyableAccessCase {
public:
    using Base = ProxyableAccessCase;
    friend class AccessCase;
    friend class InlineCacheCompiler;

    JSObject* customSlotBase() const { return m_customSlotBase.get(); }
    std::optional<DOMAttributeAnnotation> domAttribute() const { return m_domAttribute; }

    static Ref<AccessCase> create(
        VM&, JSCell* owner, AccessType, CacheableIdentifier, PropertyOffset, Structure*,
        const ObjectPropertyConditionSet&, bool viaGlobalProxy, WatchpointSet* additionalSet, CodePtr<CustomAccessorPtrTag> customGetter,
        JSObject* customSlotBase, std::optional<DOMAttributeAnnotation>, RefPtr<PolyProtoAccessChain>&&);

    static Ref<AccessCase> create(VM&, JSCell* owner, AccessType, Structure*, CacheableIdentifier, PropertyOffset,
        const ObjectPropertyConditionSet&, RefPtr<PolyProtoAccessChain>&&, bool viaGlobalProxy = false,
        CodePtr<CustomAccessorPtrTag> customSetter = nullptr, JSObject* customSlotBase = nullptr);

    CodePtr<CustomAccessorPtrTag> customAccessor() const { return m_customAccessor; }

private:
    GetterSetterAccessCase(VM&, JSCell*, AccessType, CacheableIdentifier, PropertyOffset, Structure*, const ObjectPropertyConditionSet&, bool viaGlobalProxy, WatchpointSet* additionalSet, JSObject* customSlotBase, RefPtr<PolyProtoAccessChain>&&);

    GetterSetterAccessCase(const GetterSetterAccessCase&);

    JSObject* tryGetAlternateBaseImpl() const;
    void dumpImpl(PrintStream&, CommaPrinter&, Indenter&) const;

    WriteBarrier<JSObject> m_customSlotBase;
    CodePtr<CustomAccessorPtrTag> m_customAccessor;
    std::optional<DOMAttributeAnnotation> m_domAttribute;
};

} // namespace JSC

#endif // ENABLE(JIT)
