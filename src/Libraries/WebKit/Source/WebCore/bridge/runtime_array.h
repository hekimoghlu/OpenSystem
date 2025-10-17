/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 2, 2022.
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
#ifndef RUNTIME_ARRAY_H_
#define RUNTIME_ARRAY_H_

#include "BridgeJSC.h"
#include "JSDOMBinding.h"
#include <JavaScriptCore/ArrayPrototype.h>

namespace JSC {
    
class RuntimeArray final : public JSArray {
public:
    using Base = JSArray;
    static constexpr unsigned StructureFlags = Base::StructureFlags | OverridesGetOwnPropertySlot | OverridesGetOwnPropertyNames | OverridesPut | InterceptsGetOwnPropertySlotByIndexEvenWhenLengthIsNotZero;
    static constexpr JSC::DestructionMode needsDestruction = JSC::NeedsDestruction;

    template<typename CellType, JSC::SubspaceAccess>
    static JSC::GCClient::IsoSubspace* subspaceFor(JSC::VM& vm)
    {
        return subspaceForImpl(vm);
    }

    static RuntimeArray* create(JSGlobalObject* lexicalGlobalObject, Bindings::Array* array)
    {
        VM& vm = lexicalGlobalObject->vm();
        // FIXME: deprecatedGetDOMStructure uses the prototype off of the wrong global object
        // We need to pass in the right global object for "array".
        Structure* domStructure = WebCore::deprecatedGetDOMStructure<RuntimeArray>(lexicalGlobalObject);
        RuntimeArray* runtimeArray = new (NotNull, allocateCell<RuntimeArray>(vm)) RuntimeArray(vm, domStructure);
        runtimeArray->finishCreation(vm, array);
        return runtimeArray;
    }

    typedef Bindings::Array BindingsArray;
    ~RuntimeArray();
    static void destroy(JSCell*);

    static void getOwnPropertyNames(JSObject*, JSGlobalObject*, PropertyNameArray&, DontEnumPropertiesMode);
    static bool getOwnPropertySlot(JSObject*, JSGlobalObject*, PropertyName, PropertySlot&);
    static bool getOwnPropertySlotByIndex(JSObject*, JSGlobalObject*, unsigned, PropertySlot&);
    static bool put(JSCell*, JSGlobalObject*, PropertyName, JSValue, PutPropertySlot&);
    static bool putByIndex(JSCell*, JSGlobalObject*, unsigned propertyName, JSValue, bool shouldThrow);
    
    static bool deleteProperty(JSCell*, JSGlobalObject*, PropertyName, DeletePropertySlot&);
    static bool deletePropertyByIndex(JSCell*, JSGlobalObject*, unsigned propertyName);
    
    unsigned getLength() const { return m_array->getLength(); }
    
    Bindings::Array* getConcreteArray() const { return m_array; }

    DECLARE_INFO;

    static ArrayPrototype* createPrototype(VM&, JSGlobalObject& globalObject)
    {
        return globalObject.arrayPrototype();
    }

    static Structure* createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
    {
        return Structure::create(vm, globalObject, prototype, TypeInfo(DerivedArrayType, StructureFlags), info(), NonArray);
    }

private:
    RuntimeArray(VM&, Structure*);
    void finishCreation(VM&, Bindings::Array*);

    static JSC::GCClient::IsoSubspace* subspaceForImpl(JSC::VM&);

    BindingsArray* m_array;
};

} // namespace JSC

#endif // RUNTIME_ARRAY_H_
