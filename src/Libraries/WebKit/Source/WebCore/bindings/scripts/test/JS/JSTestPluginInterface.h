/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 27, 2023.
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
#include "TestPluginInterface.h"
#include <JavaScriptCore/CallData.h>
#include <wtf/NeverDestroyed.h>

namespace WebCore {

class JSTestPluginInterface : public JSDOMWrapper<TestPluginInterface> {
public:
    using Base = JSDOMWrapper<TestPluginInterface>;
    static JSTestPluginInterface* create(JSC::Structure* structure, JSDOMGlobalObject* globalObject, Ref<TestPluginInterface>&& impl)
    {
        SUPPRESS_UNCOUNTED_LOCAL auto& vm = globalObject->vm();
        JSTestPluginInterface* ptr = new (NotNull, JSC::allocateCell<JSTestPluginInterface>(vm)) JSTestPluginInterface(structure, *globalObject, WTFMove(impl));
        ptr->finishCreation(vm);
        return ptr;
    }

    static JSC::JSObject* createPrototype(JSC::VM&, JSDOMGlobalObject&);
    static JSC::JSObject* prototype(JSC::VM&, JSDOMGlobalObject&);
    static TestPluginInterface* toWrapped(JSC::VM&, JSC::JSValue);
    static bool legacyPlatformObjectGetOwnProperty(JSC::JSObject*, JSC::JSGlobalObject*, JSC::PropertyName, JSC::PropertySlot&, bool ignoreNamedProperties);
    static bool getOwnPropertySlot(JSC::JSObject*, JSC::JSGlobalObject*, JSC::PropertyName, JSC::PropertySlot&);
    static bool getOwnPropertySlotByIndex(JSC::JSObject*, JSC::JSGlobalObject*, unsigned propertyName, JSC::PropertySlot&);
    static bool put(JSC::JSCell*, JSC::JSGlobalObject*, JSC::PropertyName, JSC::JSValue, JSC::PutPropertySlot&);
    static bool putByIndex(JSC::JSCell*, JSC::JSGlobalObject*, unsigned propertyName, JSC::JSValue, bool shouldThrow);
    static JSC::CallData getCallData(JSC::JSCell*);

    static void destroy(JSC::JSCell*);

    DECLARE_INFO;

    static JSC::Structure* createStructure(JSC::VM& vm, JSC::JSGlobalObject* globalObject, JSC::JSValue prototype)
    {
        return JSC::Structure::create(vm, globalObject, prototype, JSC::TypeInfo(JSC::ObjectType, StructureFlags), info(), JSC::MayHaveIndexedAccessors);
    }

    static JSC::JSValue getConstructor(JSC::VM&, const JSC::JSGlobalObject*);
    template<typename, JSC::SubspaceAccess mode> static JSC::GCClient::IsoSubspace* subspaceFor(JSC::VM& vm)
    {
        if constexpr (mode == JSC::SubspaceAccess::Concurrently)
            return nullptr;
        return subspaceForImpl(vm);
    }
    static JSC::GCClient::IsoSubspace* subspaceForImpl(JSC::VM& vm);
    DECLARE_VISIT_CHILDREN;

    static void analyzeHeap(JSCell*, JSC::HeapAnalyzer&);
public:
    static constexpr unsigned StructureFlags = Base::StructureFlags | JSC::InterceptsGetOwnPropertySlotByIndexEvenWhenLengthIsNotZero | JSC::OverridesGetCallData | JSC::OverridesGetOwnPropertySlot | JSC::OverridesPut | JSC::ProhibitsPropertyCaching;
protected:
    JSTestPluginInterface(JSC::Structure*, JSDOMGlobalObject&, Ref<TestPluginInterface>&&);

    DECLARE_DEFAULT_FINISH_CREATION;
};

class JSTestPluginInterfaceOwner final : public JSC::WeakHandleOwner {
public:
    bool isReachableFromOpaqueRoots(JSC::Handle<JSC::Unknown>, void* context, JSC::AbstractSlotVisitor&, ASCIILiteral*) final;
    void finalize(JSC::Handle<JSC::Unknown>, void* context) final;
};

inline JSC::WeakHandleOwner* wrapperOwner(DOMWrapperWorld&, TestPluginInterface*)
{
    static NeverDestroyed<JSTestPluginInterfaceOwner> owner;
    return &owner.get();
}

inline void* wrapperKey(TestPluginInterface* wrappableObject)
{
    return wrappableObject;
}

JSC::JSValue toJS(JSC::JSGlobalObject*, JSDOMGlobalObject*, TestPluginInterface&);
inline JSC::JSValue toJS(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, TestPluginInterface* impl) { return impl ? toJS(lexicalGlobalObject, globalObject, *impl) : JSC::jsNull(); }
JSC::JSValue toJSNewlyCreated(JSC::JSGlobalObject*, JSDOMGlobalObject*, Ref<TestPluginInterface>&&);
inline JSC::JSValue toJSNewlyCreated(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, RefPtr<TestPluginInterface>&& impl) { return impl ? toJSNewlyCreated(lexicalGlobalObject, globalObject, impl.releaseNonNull()) : JSC::jsNull(); }

template<> struct JSDOMWrapperConverterTraits<TestPluginInterface> {
    using WrapperClass = JSTestPluginInterface;
    using ToWrappedReturnType = TestPluginInterface*;
};

} // namespace WebCore
