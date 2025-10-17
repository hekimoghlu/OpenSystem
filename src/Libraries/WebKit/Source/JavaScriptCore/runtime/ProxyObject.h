/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 13, 2022.
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

#include "JSInternalFieldObjectImpl.h"

namespace JSC {

class ProxyObject final : public JSInternalFieldObjectImpl<2> {
public:
    using Base = JSInternalFieldObjectImpl<2>;

    static constexpr unsigned StructureFlags = Base::StructureFlags | OverridesGetOwnPropertySlot | OverridesGetOwnPropertyNames | OverridesGetPrototype | OverridesGetCallData | OverridesPut | OverridesIsExtensible | InterceptsGetOwnPropertySlotByIndexEvenWhenLengthIsNotZero | ProhibitsPropertyCaching;

    enum class Field : uint32_t {
        Target = 0,
        Handler,
    };
    static_assert(numberOfInternalFields == 2);
    static std::array<JSValue, numberOfInternalFields> initialValues()
    {
        return { {
            jsNull(),
            jsUndefined(),
        } };
    }

    enum class HandlerTrap : uint8_t {
        Has = 0,
        Get,
        GetOwnPropertyDescriptor,
        OwnKeys,
        Set,
    };

    static constexpr unsigned numberOfCachedHandlerTrapsOffsets = 5;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.proxyObjectSpace<mode>();
    }

    static ProxyObject* create(JSGlobalObject* globalObject, JSValue target, JSValue handler)
    {
        VM& vm = getVM(globalObject);
        Structure* structure = ProxyObject::structureForTarget(globalObject, target);
        ProxyObject* proxy = new (NotNull, allocateCell<ProxyObject>(vm)) ProxyObject(vm, structure);
        proxy->finishCreation(vm, globalObject, target, handler);
        return proxy;
    }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue, bool);

    DECLARE_EXPORT_INFO;

    JSObject* target() const { return jsCast<JSObject*>(internalField(Field::Target).get()); }
    JSValue handler() const { return internalField(Field::Handler).get(); }

    static void validateNegativeHasTrapResult(JSGlobalObject*, JSObject*, PropertyName);
    static void validatePositiveSetTrapResult(JSGlobalObject*, JSObject*, PropertyName, JSValue putValue);
    static void validateGetTrapResult(JSGlobalObject*, JSValue trapResult, JSObject*, PropertyName);

    static bool put(JSCell*, JSGlobalObject*, PropertyName, JSValue, PutPropertySlot&);
    static bool putByIndex(JSCell*, JSGlobalObject*, unsigned propertyName, JSValue, bool shouldThrow);
    bool putByIndexCommon(JSGlobalObject*, JSValue thisValue, unsigned propertyName, JSValue putValue, bool shouldThrow);
    JSValue performGetPrototype(JSGlobalObject*);
    void revoke(VM&);
    bool isRevoked() const;

    const WriteBarrier<Unknown>& internalField(Field field) const { return Base::internalField(static_cast<uint32_t>(field)); }
    WriteBarrier<Unknown>& internalField(Field field) { return Base::internalField(static_cast<uint32_t>(field)); }

    JSObject* getHandlerTrap(JSGlobalObject*, JSObject*, CallData&, const Identifier&, HandlerTrap);
    bool forwardsGetOwnPropertyNamesToTarget(DontEnumPropertiesMode);

private:
    JS_EXPORT_PRIVATE ProxyObject(VM&, Structure*);
    JS_EXPORT_PRIVATE void finishCreation(VM&, JSGlobalObject*, JSValue target, JSValue handler);
    JS_EXPORT_PRIVATE static Structure* structureForTarget(JSGlobalObject*, JSValue target);

    bool isHandlerTrapsCacheValid(JSObject* handler);
    void clearHandlerTrapsOffsetsCache();

    static bool getOwnPropertySlot(JSObject*, JSGlobalObject*, PropertyName, PropertySlot&);
    static bool getOwnPropertySlotByIndex(JSObject*, JSGlobalObject*, unsigned propertyName, PropertySlot&);
    static CallData getCallData(JSCell*);
    static CallData getConstructData(JSCell*);
    static bool deleteProperty(JSCell*, JSGlobalObject*, PropertyName, DeletePropertySlot&);
    static bool deletePropertyByIndex(JSCell*, JSGlobalObject*, unsigned propertyName);
    static bool preventExtensions(JSObject*, JSGlobalObject*);
    static bool isExtensible(JSObject*, JSGlobalObject*);
    static bool defineOwnProperty(JSObject*, JSGlobalObject*, PropertyName, const PropertyDescriptor&, bool shouldThrow);
    static void getOwnPropertyNames(JSObject*, JSGlobalObject*, PropertyNameArray&, DontEnumPropertiesMode);
    static bool setPrototype(JSObject*, JSGlobalObject*, JSValue prototype, bool shouldThrowIfCantSet);
    static JSValue getPrototype(JSObject*, JSGlobalObject*);
    DECLARE_VISIT_CHILDREN;

    bool getOwnPropertySlotCommon(JSGlobalObject*, PropertyName, PropertySlot&);
    bool performInternalMethodGetOwnProperty(JSGlobalObject*, PropertyName, PropertySlot&);
    bool performGet(JSGlobalObject*, PropertyName, PropertySlot&);
    bool performHasProperty(JSGlobalObject*, PropertyName, PropertySlot&);
    template <typename DefaultDeleteFunction>
    bool performDelete(JSGlobalObject*, PropertyName, DefaultDeleteFunction);
    template <typename PerformDefaultPutFunction>
    bool performPut(JSGlobalObject*, JSValue putValue, JSValue thisValue, PropertyName, PerformDefaultPutFunction, bool shouldThrow);
    bool performPreventExtensions(JSGlobalObject*);
    bool performIsExtensible(JSGlobalObject*);
    bool performDefineOwnProperty(JSGlobalObject*, PropertyName, const PropertyDescriptor&, bool shouldThrow);
    void performGetOwnPropertyNames(JSGlobalObject*, PropertyNameArray&);
    void performGetOwnEnumerablePropertyNames(JSGlobalObject*, PropertyNameArray&);
    bool performSetPrototype(JSGlobalObject*, JSValue prototype, bool shouldThrowIfCantSet);

    bool m_isCallable : 1 { false };
    bool m_isConstructible : 1 { false };
    PropertyOffset m_handlerTrapsOffsetsCache[numberOfCachedHandlerTrapsOffsets];
    WriteBarrierStructureID m_handlerStructureID;
    WriteBarrierStructureID m_handlerPrototypeStructureID;
};

ALWAYS_INLINE bool ProxyObject::isHandlerTrapsCacheValid(JSObject* handler)
{
    return handler->structureID() == m_handlerStructureID.value() && asObject(handler->getPrototypeDirect())->structureID() == m_handlerPrototypeStructureID.value();
}

} // namespace JSC
