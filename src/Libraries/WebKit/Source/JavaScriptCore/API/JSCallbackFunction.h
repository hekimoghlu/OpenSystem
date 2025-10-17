/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 25, 2023.
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
#ifndef JSCallbackFunction_h
#define JSCallbackFunction_h

#include "InternalFunction.h"
#include "JSObjectRef.h"

namespace JSC {

#define JSCALLBACK_FUNCTION_METHOD(method) \
    WTF_VTBL_FUNCPTR_PTRAUTH_STR("JSCallbackFunction." #method) method

class JSCallbackFunction final : public InternalFunction {
    friend struct APICallbackFunction;
public:
    typedef InternalFunction Base;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.callbackFunctionSpace<mode>();
    }

    static JSCallbackFunction* create(VM&, JSGlobalObject*, JSObjectCallAsFunctionCallback, const String& name);

    DECLARE_INFO;
    
    // InternalFunction mish-mashes constructor and function behavior -- we should 
    // refactor the code so this override isn't necessary
    static Structure* createStructure(VM& vm, JSGlobalObject* globalObject, JSValue proto) 
    { 
        return Structure::create(vm, globalObject, proto, TypeInfo(InternalFunctionType, StructureFlags), info()); 
    }

private:
    JSCallbackFunction(VM&, Structure*, JSObjectCallAsFunctionCallback);
    void finishCreation(VM&, const String& name);

    JSObjectCallAsFunctionCallback functionCallback() { return m_callback; }

    JSObjectCallAsFunctionCallback JSCALLBACK_FUNCTION_METHOD(m_callback) { nullptr };
};

#undef JSCALLBACK_FUNCTION_METHOD

} // namespace JSC

#endif // JSCallbackFunction_h
