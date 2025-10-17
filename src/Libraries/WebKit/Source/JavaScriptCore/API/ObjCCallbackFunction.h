/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 10, 2024.
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
#ifndef ObjCCallbackFunction_h
#define ObjCCallbackFunction_h 

#include <JavaScriptCore/JSBase.h>

#if JSC_OBJC_API_ENABLED

#import "JSCallbackFunction.h"

#if defined(__OBJC__)
@class JSContext;

JSObjectRef objCCallbackFunctionForMethod(JSContext *, Class, Protocol *, BOOL isInstanceMethod, SEL, const char* types);
JSObjectRef objCCallbackFunctionForBlock(JSContext *, id);
JSObjectRef objCCallbackFunctionForInit(JSContext *, Class, Protocol *, SEL, const char* types);

id tryUnwrapConstructor(JSObjectRef);
#endif

namespace JSC {

class ObjCCallbackFunctionImpl;

#define OBJC_CALLBACK_FUNCTION_METHOD(method) \
    WTF_VTBL_FUNCPTR_PTRAUTH_STR("ObjCCallbackFunction." #method) method

class ObjCCallbackFunction : public InternalFunction {
    friend struct APICallbackFunction;
public:
    typedef InternalFunction Base;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.objCCallbackFunctionSpace<mode>();
    }

    static ObjCCallbackFunction* create(VM&, JSGlobalObject*, const String& name, std::unique_ptr<ObjCCallbackFunctionImpl>);
    static constexpr DestructionMode needsDestruction = NeedsDestruction;
    static void destroy(JSCell*);

    static Structure* createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
    {
        ASSERT(globalObject);
        return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
    }

    DECLARE_EXPORT_INFO;

    ObjCCallbackFunctionImpl* impl() const { return m_impl.get(); }

protected:
    ObjCCallbackFunction(VM&, Structure*, JSObjectCallAsFunctionCallback, JSObjectCallAsConstructorCallback, std::unique_ptr<ObjCCallbackFunctionImpl>);

private:
    JSObjectCallAsFunctionCallback functionCallback() { return m_functionCallback; }
    JSObjectCallAsConstructorCallback constructCallback() { return m_constructCallback; }

    JSObjectCallAsFunctionCallback OBJC_CALLBACK_FUNCTION_METHOD(m_functionCallback);
    JSObjectCallAsConstructorCallback OBJC_CALLBACK_FUNCTION_METHOD(m_constructCallback);
    std::unique_ptr<ObjCCallbackFunctionImpl> m_impl;
};

#undef OBJC_CALLBACK_FUNCTION_METHOD

} // namespace JSC

#endif

#endif // ObjCCallbackFunction_h 
