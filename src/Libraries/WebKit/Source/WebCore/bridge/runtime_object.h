/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 27, 2023.
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
#ifndef KJS_RUNTIME_OBJECT_H
#define KJS_RUNTIME_OBJECT_H

#include "BridgeJSC.h"
#include <JavaScriptCore/JSGlobalObject.h>

namespace JSC {
namespace Bindings {

Exception* throwRuntimeObjectInvalidAccessError(JSGlobalObject*, ThrowScope&);

class WEBCORE_EXPORT RuntimeObject : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;
    static constexpr unsigned StructureFlags = Base::StructureFlags | OverridesGetOwnPropertySlot | OverridesGetOwnPropertyNames | OverridesGetCallData | OverridesPut | GetOwnPropertySlotMayBeWrongAboutDontEnum;
    static constexpr JSC::DestructionMode needsDestruction = JSC::NeedsDestruction;

    template<typename CellType, JSC::SubspaceAccess>
    static GCClient::IsoSubspace* subspaceFor(JSC::VM& vm)
    {
        static_assert(sizeof(CellType) == sizeof(RuntimeObject), "RuntimeObject subclasses that add fields need to override subspaceFor<>()");
        static_assert(CellType::destroy == RuntimeObject::destroy);
        return subspaceForImpl(vm);
    }

    static RuntimeObject* create(VM& vm, Structure* structure, RefPtr<Instance>&& instance)
    {
        RuntimeObject* object = new (NotNull, allocateCell<RuntimeObject>(vm)) RuntimeObject(vm, structure, WTFMove(instance));
        object->finishCreation(vm);
        return object;
    }

    static void destroy(JSCell*);

    static bool getOwnPropertySlot(JSObject*, JSGlobalObject*, PropertyName, PropertySlot&);
    static bool put(JSCell*, JSGlobalObject*, PropertyName, JSValue, PutPropertySlot&);
    static bool deleteProperty(JSCell*, JSGlobalObject*, PropertyName, DeletePropertySlot&);
    static CallData getCallData(JSCell*);
    static CallData getConstructData(JSCell*);

    static void getOwnPropertyNames(JSObject*, JSGlobalObject*, PropertyNameArray&, DontEnumPropertiesMode);

    void invalidate();

    Instance* getInternalInstance() const { return m_instance.get(); }

    DECLARE_INFO;

    static ObjectPrototype* createPrototype(VM&, JSGlobalObject& globalObject)
    {
        return globalObject.objectPrototype();
    }

    static Structure* createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
    {
        return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
    }

protected:
    RuntimeObject(VM&, Structure*, RefPtr<Instance>&&);
    void finishCreation(VM&);

private:
    static GCClient::IsoSubspace* subspaceForImpl(VM&);

    RefPtr<Instance> m_instance;
};
    
}
}

#endif
