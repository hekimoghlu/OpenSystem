/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 16, 2024.
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
#ifndef JSCallbackConstructor_h
#define JSCallbackConstructor_h

#include "JSObject.h"
#include "JSObjectRef.h"

namespace JSC {

#define JSCALLBACK_CONSTRUCTOR_METHOD(method) \
    WTF_VTBL_FUNCPTR_PTRAUTH_STR("JSCallbackConstructor." #method) method

class JSCallbackConstructor final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;
    static constexpr unsigned StructureFlags = Base::StructureFlags | ImplementsHasInstance | ImplementsDefaultHasInstance;
    static constexpr DestructionMode needsDestruction = NeedsDestruction;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.callbackConstructorSpace<mode>();
    }

    static JSCallbackConstructor* create(JSGlobalObject* globalObject, Structure* structure, JSClassRef classRef, JSObjectCallAsConstructorCallback callback)
    {
        VM& vm = getVM(globalObject);
        JSCallbackConstructor* constructor = new (NotNull, allocateCell<JSCallbackConstructor>(vm)) JSCallbackConstructor(globalObject, structure, classRef, callback);
        constructor->finishCreation(globalObject, classRef);
        return constructor;
    }
    
    ~JSCallbackConstructor();
    static void destroy(JSCell*);
    JSClassRef classRef() const { return m_class; }
    JSObjectCallAsConstructorCallback callback() const { return m_callback; }
    DECLARE_INFO;

    static Structure* createStructure(VM& vm, JSGlobalObject* globalObject, JSValue proto)
    {
        return Structure::create(vm, globalObject, proto, TypeInfo(ObjectType, StructureFlags), info());
    }

private:
    JSCallbackConstructor(JSGlobalObject*, Structure*, JSClassRef, JSObjectCallAsConstructorCallback);
    void finishCreation(JSGlobalObject*, JSClassRef);

    friend struct APICallbackFunction;

    static CallData getConstructData(JSCell*);

    JSObjectCallAsConstructorCallback constructCallback() { return m_callback; }

    JSClassRef m_class;
    JSObjectCallAsConstructorCallback JSCALLBACK_CONSTRUCTOR_METHOD(m_callback);
};

#undef JSCALLBACK_CONSTRUCTOR_METHOD

} // namespace JSC

#endif // JSCallbackConstructor_h
