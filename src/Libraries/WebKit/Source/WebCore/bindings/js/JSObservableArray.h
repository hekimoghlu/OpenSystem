/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 19, 2024.
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

#include "JSDOMBinding.h"
#include <JavaScriptCore/ArrayPrototype.h>

namespace JSC {

class ObservableArray : public RefCounted<ObservableArray> {
public:
    virtual ~ObservableArray() { }

    virtual bool setValueAt(JSGlobalObject*, unsigned index, JSValue) = 0;
    virtual void removeLast() = 0;
    virtual JSValue valueAt(JSGlobalObject*, unsigned index) const = 0;
    virtual unsigned length() const = 0;
    virtual void shrinkTo(unsigned) = 0;
};

class JSObservableArray final : public JSArray {
public:
    using Base = JSArray;
    static constexpr unsigned StructureFlags = Base::StructureFlags | OverridesGetOwnPropertySlot | OverridesGetOwnPropertyNames | OverridesPut | InterceptsGetOwnPropertySlotByIndexEvenWhenLengthIsNotZero;
    static constexpr JSC::DestructionMode needsDestruction = JSC::NeedsDestruction;

    template<typename CellType, JSC::SubspaceAccess>
    static JSC::GCClient::IsoSubspace* subspaceFor(JSC::VM& vm)
    {
        return subspaceForImpl(vm);
    }

    static JSObservableArray* create(JSGlobalObject* lexicalGlobalObject, Ref<ObservableArray>&& array)
    {
        VM& vm = lexicalGlobalObject->vm();
        // FIXME: deprecatedGetDOMStructure uses the prototype off of the wrong global object
        // We need to pass in the right global object for "array".
        Structure* domStructure = WebCore::deprecatedGetDOMStructure<JSObservableArray>(lexicalGlobalObject);
        JSObservableArray* observableArray = new (NotNull, allocateCell<JSObservableArray>(vm)) JSObservableArray(vm, domStructure);
        observableArray->finishCreation(vm, WTFMove(array));
        return observableArray;
    }

    ~JSObservableArray();
    static void destroy(JSCell*);

    static bool defineOwnProperty(JSObject*, JSGlobalObject*, PropertyName, const PropertyDescriptor&, bool throwException);
    static void getOwnPropertyNames(JSObject*, JSGlobalObject*, PropertyNameArray&, DontEnumPropertiesMode);
    static bool getOwnPropertySlot(JSObject*, JSGlobalObject*, PropertyName, PropertySlot&);
    static bool getOwnPropertySlotByIndex(JSObject*, JSGlobalObject*, unsigned, PropertySlot&);
    static bool put(JSCell*, JSGlobalObject*, PropertyName, JSValue, PutPropertySlot&);
    static bool putByIndex(JSCell*, JSGlobalObject*, unsigned propertyName, JSValue, bool shouldThrow);

    static bool deleteProperty(JSCell*, JSGlobalObject*, PropertyName, DeletePropertySlot&);
    static bool deletePropertyByIndex(JSCell*, JSGlobalObject*, unsigned propertyName);

    unsigned length() const { return m_array->length(); }

    ObservableArray& getConcreteArray() const { return *m_array; }

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
    JSObservableArray(VM&, Structure*);
    void finishCreation(VM&, Ref<ObservableArray>&&);

    static JSC::GCClient::IsoSubspace* subspaceForImpl(JSC::VM&);

    RefPtr<ObservableArray> m_array;
};

} // namespace JSC
