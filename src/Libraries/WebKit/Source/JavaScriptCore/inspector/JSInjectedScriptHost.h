/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 27, 2024.
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

namespace Inspector {

class InjectedScriptHost;

class JSInjectedScriptHost final : public JSC::JSNonFinalObject {
public:
    using Base = JSC::JSNonFinalObject;
    static constexpr unsigned StructureFlags = Base::StructureFlags;
    static constexpr JSC::DestructionMode needsDestruction = JSC::NeedsDestruction;

    template<typename CellType, JSC::SubspaceAccess mode>
    static JSC::GCClient::IsoSubspace* subspaceFor(JSC::VM& vm)
    {
        return vm.injectedScriptHostSpace<mode>();
    }

    DECLARE_INFO;

    static JSC::Structure* createStructure(JSC::VM& vm, JSC::JSGlobalObject* globalObject, JSC::JSValue prototype)
    {
        return JSC::Structure::create(vm, globalObject, prototype, JSC::TypeInfo(JSC::ObjectType, StructureFlags), info());
    }

    static JSInjectedScriptHost* create(JSC::VM& vm, JSC::Structure* structure, Ref<InjectedScriptHost>&& impl)
    {
        JSInjectedScriptHost* instance = new (NotNull, JSC::allocateCell<JSInjectedScriptHost>(vm)) JSInjectedScriptHost(vm, structure, WTFMove(impl));
        instance->finishCreation(vm);
        return instance;
    }

    static JSC::JSObject* createPrototype(JSC::VM&, JSC::JSGlobalObject*);
    static void destroy(JSC::JSCell*);

    InjectedScriptHost& impl() const { return m_wrapped; }

    // Attributes.
    JSC::JSValue evaluate(JSC::JSGlobalObject*) const;
    JSC::JSValue savedResultAlias(JSC::JSGlobalObject*) const;

    // Functions.
    JSC::JSValue evaluateWithScopeExtension(JSC::JSGlobalObject*, JSC::CallFrame*);
    JSC::JSValue internalConstructorName(JSC::JSGlobalObject*, JSC::CallFrame*);
    JSC::JSValue isHTMLAllCollection(JSC::JSGlobalObject*, JSC::CallFrame*);
    JSC::JSValue isPromiseRejectedWithNativeGetterTypeError(JSC::JSGlobalObject*, JSC::CallFrame*);
    JSC::JSValue subtype(JSC::JSGlobalObject*, JSC::CallFrame*);
    JSC::JSValue functionDetails(JSC::JSGlobalObject*, JSC::CallFrame*);
    JSC::JSValue getOwnPrivatePropertySymbols(JSC::JSGlobalObject*, JSC::CallFrame*);
    JSC::JSValue getInternalProperties(JSC::JSGlobalObject*, JSC::CallFrame*);
    JSC::JSValue proxyTargetValue(JSC::CallFrame*);
    JSC::JSValue weakRefTargetValue(JSC::JSGlobalObject*, JSC::CallFrame*);
    JSC::JSValue weakMapSize(JSC::JSGlobalObject*, JSC::CallFrame*);
    JSC::JSValue weakMapEntries(JSC::JSGlobalObject*, JSC::CallFrame*);
    JSC::JSValue weakSetSize(JSC::JSGlobalObject*, JSC::CallFrame*);
    JSC::JSValue weakSetEntries(JSC::JSGlobalObject*, JSC::CallFrame*);
    JSC::JSValue iteratorEntries(JSC::JSGlobalObject*, JSC::CallFrame*);
    JSC::JSValue queryInstances(JSC::JSGlobalObject*, JSC::CallFrame*);
    JSC::JSValue queryHolders(JSC::JSGlobalObject*, JSC::CallFrame*);

private:
    JSInjectedScriptHost(JSC::VM&, JSC::Structure*, Ref<InjectedScriptHost>&&);
    DECLARE_DEFAULT_FINISH_CREATION;

    Ref<InjectedScriptHost> m_wrapped;
};

} // namespace Inspector
