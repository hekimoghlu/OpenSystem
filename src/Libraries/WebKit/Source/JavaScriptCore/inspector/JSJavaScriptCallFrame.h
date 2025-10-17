/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 7, 2024.
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
#include "JavaScriptCallFrame.h"

namespace Inspector {

class JSJavaScriptCallFrame final : public JSC::JSNonFinalObject {
public:
    using Base = JSC::JSNonFinalObject;
    static constexpr unsigned StructureFlags = Base::StructureFlags;
    static constexpr JSC::DestructionMode needsDestruction = JSC::NeedsDestruction;

    template<typename CellType, JSC::SubspaceAccess mode>
    static JSC::GCClient::IsoSubspace* subspaceFor(JSC::VM& vm)
    {
        return vm.javaScriptCallFrameSpace<mode>();
    }

    DECLARE_INFO;

    static JSC::Structure* createStructure(JSC::VM& vm, JSC::JSGlobalObject* globalObject, JSC::JSValue prototype)
    {
        return JSC::Structure::create(vm, globalObject, prototype, JSC::TypeInfo(JSC::ObjectType, StructureFlags), info());
    }

    static JSJavaScriptCallFrame* create(JSC::VM& vm, JSC::Structure* structure, Ref<JavaScriptCallFrame>&& impl)
    {
        JSJavaScriptCallFrame* instance = new (NotNull, JSC::allocateCell<JSJavaScriptCallFrame>(vm)) JSJavaScriptCallFrame(vm, structure, WTFMove(impl));
        instance->finishCreation(vm);
        return instance;
    }

    static JSC::JSObject* createPrototype(JSC::VM&, JSC::JSGlobalObject*);
    static void destroy(JSC::JSCell*);

    JavaScriptCallFrame& impl() const { return *m_impl; }
    void releaseImpl();

    // Functions.
    JSC::JSValue evaluateWithScopeExtension(JSC::JSGlobalObject*, JSC::CallFrame*);
    JSC::JSValue scopeDescriptions(JSC::JSGlobalObject*);

    // Attributes.
    JSC::JSValue caller(JSC::JSGlobalObject*) const;
    JSC::JSValue sourceID(JSC::JSGlobalObject*) const;
    JSC::JSValue line(JSC::JSGlobalObject*) const;
    JSC::JSValue column(JSC::JSGlobalObject*) const;
    JSC::JSValue functionName(JSC::JSGlobalObject*) const;
    JSC::JSValue scopeChain(JSC::JSGlobalObject*) const;
    JSC::JSValue thisObject(JSC::JSGlobalObject*) const;
    JSC::JSValue type(JSC::JSGlobalObject*) const;
    JSC::JSValue isTailDeleted(JSC::JSGlobalObject*) const;

    // Constants.
    static constexpr unsigned short GLOBAL_SCOPE = 0;
    static constexpr unsigned short WITH_SCOPE = 1;
    static constexpr unsigned short CLOSURE_SCOPE = 2;
    static constexpr unsigned short CATCH_SCOPE = 3;
    static constexpr unsigned short FUNCTION_NAME_SCOPE = 4;
    static constexpr unsigned short GLOBAL_LEXICAL_ENVIRONMENT_SCOPE = 5;
    static constexpr unsigned short NESTED_LEXICAL_SCOPE = 6;

private:
    JSJavaScriptCallFrame(JSC::VM&, JSC::Structure*, Ref<JavaScriptCallFrame>&&);
    ~JSJavaScriptCallFrame();
    DECLARE_DEFAULT_FINISH_CREATION;

    JavaScriptCallFrame* m_impl;
};

JSC::JSValue toJS(JSC::JSGlobalObject*, JSC::JSGlobalObject*, JavaScriptCallFrame*);

} // namespace Inspector
