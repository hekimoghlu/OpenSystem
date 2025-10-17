/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 1, 2024.
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

#include "InternalFunction.h"
#include "JSObjectRef.h"
#include <glib-object.h>
#include <wtf/glib/GRefPtr.h>

typedef struct _JSCClass JSCClass;

namespace JSC {

class JSCCallbackFunction final : public InternalFunction {
    friend struct APICallbackFunction;
public:
    typedef InternalFunction Base;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.jscCallbackFunctionSpace<mode>();
    }

    enum class Type {
        Function,
        Method,
        Constructor
    };

    static JSCCallbackFunction* create(VM&, JSGlobalObject*, const String& name, Type, JSCClass*, GRefPtr<GClosure>&&, GType, std::optional<Vector<GType>>&&);
    static constexpr DestructionMode needsDestruction = NeedsDestruction;
    static void destroy(JSCell*);

    static Structure* createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
    {
        ASSERT(globalObject);
        return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
    }

    DECLARE_INFO;

    JSValueRef call(JSContextRef, JSObjectRef, size_t argumentCount, const JSValueRef arguments[], JSValueRef* exception);
    JSObjectRef construct(JSContextRef, size_t argumentCount, const JSValueRef arguments[], JSValueRef* exception);

private:
    JSCCallbackFunction(VM&, Structure*, Type, JSCClass*, GRefPtr<GClosure>&&, GType, std::optional<Vector<GType>>&&);

    JSObjectCallAsFunctionCallback functionCallback() { return m_functionCallback; }
    JSObjectCallAsConstructorCallback constructCallback() { return m_constructCallback; }

    JSObjectCallAsFunctionCallback m_functionCallback;
    JSObjectCallAsConstructorCallback m_constructCallback;
    Type m_type;
    GRefPtr<JSCClass> m_class;
    GRefPtr<GClosure> m_closure;
    GType m_returnType;
    std::optional<Vector<GType>> m_parameters;
};

} // namespace JSC
