/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 2, 2021.
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

#include "JSObject.h"

namespace JSC {

class JSGlobalProxy : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;
    static constexpr unsigned StructureFlags = Base::StructureFlags | OverridesGetOwnPropertySlot | OverridesGetOwnPropertyNames | OverridesPut | OverridesGetPrototype | OverridesIsExtensible | InterceptsGetOwnPropertySlotByIndexEvenWhenLengthIsNotZero;

    template<typename CellType, SubspaceAccess>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        static_assert(sizeof(CellType) == sizeof(JSGlobalProxy));
        return &vm.jsGlobalProxySpace();
    }

    static JSGlobalProxy* create(VM& vm, Structure* structure)
    {
        JSGlobalProxy* proxy = new (NotNull, allocateCell<JSGlobalProxy>(vm)) JSGlobalProxy(vm, structure, nullptr);
        proxy->finishCreation(vm);
        return proxy;
    }

    static JSGlobalProxy* create(VM& vm, Structure* structure, JSGlobalObject* globalObject)
    {
        JSGlobalProxy* proxy = new (NotNull, allocateCell<JSGlobalProxy>(vm)) JSGlobalProxy(vm, structure, globalObject);
        proxy->finishCreation(vm);
        return proxy;
    }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_EXPORT_INFO;

    JSGlobalObject* target() const { return m_target.get(); }
    static constexpr ptrdiff_t targetOffset() { return OBJECT_OFFSETOF(JSGlobalProxy, m_target); }

    JS_EXPORT_PRIVATE void setTarget(VM&, JSGlobalObject*);

protected:
    JSGlobalProxy(VM& vm, Structure* structure, JSGlobalObject* target)
        : Base(vm, structure)
        , m_target(target, WriteBarrierEarlyInit)
    {
    }

    DECLARE_DEFAULT_FINISH_CREATION;

    DECLARE_VISIT_CHILDREN_WITH_MODIFIER(JS_EXPORT_PRIVATE);

    JS_EXPORT_PRIVATE static bool getOwnPropertySlot(JSObject*, JSGlobalObject*, PropertyName, PropertySlot&);
    JS_EXPORT_PRIVATE static bool getOwnPropertySlotByIndex(JSObject*, JSGlobalObject*, unsigned, PropertySlot&);
    JS_EXPORT_PRIVATE static bool put(JSCell*, JSGlobalObject*, PropertyName, JSValue, PutPropertySlot&);
    JS_EXPORT_PRIVATE static bool putByIndex(JSCell*, JSGlobalObject*, unsigned, JSValue, bool shouldThrow);
    JS_EXPORT_PRIVATE static bool deleteProperty(JSCell*, JSGlobalObject*, PropertyName, DeletePropertySlot&);
    JS_EXPORT_PRIVATE static bool deletePropertyByIndex(JSCell*, JSGlobalObject*, unsigned);
    JS_EXPORT_PRIVATE static void getOwnPropertyNames(JSObject*, JSGlobalObject*, PropertyNameArray&, DontEnumPropertiesMode);
    JS_EXPORT_PRIVATE static bool defineOwnProperty(JSObject*, JSGlobalObject*, PropertyName, const PropertyDescriptor&, bool shouldThrow);
    JS_EXPORT_PRIVATE static bool setPrototype(JSObject*, JSGlobalObject*, JSValue, bool shouldThrowIfCantSet);
    JS_EXPORT_PRIVATE static JSValue getPrototype(JSObject*, JSGlobalObject*);
    JS_EXPORT_PRIVATE static bool isExtensible(JSObject*, JSGlobalObject*);
    JS_EXPORT_PRIVATE static bool preventExtensions(JSObject*, JSGlobalObject*);

private:
    WriteBarrier<JSGlobalObject> m_target;
};

} // namespace JSC
