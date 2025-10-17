/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 9, 2023.
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

#include "JSDOMWrapper.h"

namespace WebCore {

class JSDOMWindowProperties final : public JSDOMObject {
public:
    using Base = JSDOMObject;
    static constexpr unsigned StructureFlags = Base::StructureFlags | JSC::GetOwnPropertySlotIsImpureForPropertyAbsence | JSC::InterceptsGetOwnPropertySlotByIndexEvenWhenLengthIsNotZero | JSC::OverridesGetOwnPropertySlot | JSC::OverridesIsExtensible | JSC::IsImmutablePrototypeExoticObject;

    static constexpr JSC::DestructionMode needsDestruction = JSC::DoesNotNeedDestruction;
    template<typename CellType, JSC::SubspaceAccess>
    static JSC::GCClient::IsoSubspace* subspaceFor(JSC::VM& vm)
    {
        static_assert(CellType::destroy == JSC::JSCell::destroy, "JSDOMWindowProperties is not destructible actually");
        return subspaceForImpl(vm);
    }

    static JSDOMWindowProperties* create(JSC::Structure* structure, JSC::JSGlobalObject& globalObject)
    {
        JSDOMWindowProperties* ptr = new (NotNull, JSC::allocateCell<JSDOMWindowProperties>(globalObject.vm())) JSDOMWindowProperties(structure, globalObject);
        ptr->finishCreation(globalObject);
        return ptr;
    }

    DECLARE_INFO;

    static JSC::Structure* createStructure(JSC::VM& vm, JSC::JSGlobalObject* globalObject, JSC::JSValue prototype)
    {
        return JSC::Structure::create(vm, globalObject, prototype, JSC::TypeInfo(JSC::ObjectType, StructureFlags), info(), JSC::MayHaveIndexedAccessors);
    }

    static bool getOwnPropertySlot(JSC::JSObject*, JSC::JSGlobalObject*, JSC::PropertyName, JSC::PropertySlot&);
    static bool getOwnPropertySlotByIndex(JSC::JSObject*, JSC::JSGlobalObject*, unsigned propertyName, JSC::PropertySlot&);
    static bool deleteProperty(JSC::JSCell*, JSC::JSGlobalObject*, JSC::PropertyName, JSC::DeletePropertySlot&);
    static bool deletePropertyByIndex(JSC::JSCell*, JSC::JSGlobalObject*, unsigned propertyName);
    static bool preventExtensions(JSC::JSObject*, JSC::JSGlobalObject*);
    static bool isExtensible(JSC::JSObject*, JSC::JSGlobalObject*);
    static bool defineOwnProperty(JSC::JSObject*, JSC::JSGlobalObject*, JSC::PropertyName, const JSC::PropertyDescriptor&, bool shouldThrow);

private:
    JSDOMWindowProperties(JSC::Structure* structure, JSC::JSGlobalObject& globalObject)
        : JSDOMObject(structure, globalObject)
    {
    }

    void finishCreation(JSC::JSGlobalObject&);
    static JSC::GCClient::IsoSubspace* subspaceForImpl(JSC::VM&);
};

} // namespace WebCore
