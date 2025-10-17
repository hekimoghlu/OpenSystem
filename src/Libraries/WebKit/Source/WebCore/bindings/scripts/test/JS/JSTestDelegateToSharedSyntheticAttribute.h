/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 31, 2023.
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
#include "TestDelegateToSharedSyntheticAttribute.h"
#include <wtf/NeverDestroyed.h>

namespace WebCore {

class JSTestDelegateToSharedSyntheticAttribute : public JSDOMWrapper<TestDelegateToSharedSyntheticAttribute> {
public:
    using Base = JSDOMWrapper<TestDelegateToSharedSyntheticAttribute>;
    static JSTestDelegateToSharedSyntheticAttribute* create(JSC::Structure* structure, JSDOMGlobalObject* globalObject, Ref<TestDelegateToSharedSyntheticAttribute>&& impl)
    {
        SUPPRESS_UNCOUNTED_LOCAL auto& vm = globalObject->vm();
        JSTestDelegateToSharedSyntheticAttribute* ptr = new (NotNull, JSC::allocateCell<JSTestDelegateToSharedSyntheticAttribute>(vm)) JSTestDelegateToSharedSyntheticAttribute(structure, *globalObject, WTFMove(impl));
        ptr->finishCreation(vm);
        return ptr;
    }

    static JSC::JSObject* createPrototype(JSC::VM&, JSDOMGlobalObject&);
    static JSC::JSObject* prototype(JSC::VM&, JSDOMGlobalObject&);
    static TestDelegateToSharedSyntheticAttribute* toWrapped(JSC::VM&, JSC::JSValue);
    static void destroy(JSC::JSCell*);

    DECLARE_INFO;

    static JSC::Structure* createStructure(JSC::VM& vm, JSC::JSGlobalObject* globalObject, JSC::JSValue prototype)
    {
        return JSC::Structure::create(vm, globalObject, prototype, JSC::TypeInfo(JSC::ObjectType, StructureFlags), info(), JSC::NonArray);
    }

    static JSC::JSValue getConstructor(JSC::VM&, const JSC::JSGlobalObject*);
    template<typename, JSC::SubspaceAccess mode> static JSC::GCClient::IsoSubspace* subspaceFor(JSC::VM& vm)
    {
        if constexpr (mode == JSC::SubspaceAccess::Concurrently)
            return nullptr;
        return subspaceForImpl(vm);
    }
    static JSC::GCClient::IsoSubspace* subspaceForImpl(JSC::VM& vm);
    static void analyzeHeap(JSCell*, JSC::HeapAnalyzer&);
protected:
    JSTestDelegateToSharedSyntheticAttribute(JSC::Structure*, JSDOMGlobalObject&, Ref<TestDelegateToSharedSyntheticAttribute>&&);

    DECLARE_DEFAULT_FINISH_CREATION;
};

class JSTestDelegateToSharedSyntheticAttributeOwner final : public JSC::WeakHandleOwner {
public:
    bool isReachableFromOpaqueRoots(JSC::Handle<JSC::Unknown>, void* context, JSC::AbstractSlotVisitor&, ASCIILiteral*) final;
    void finalize(JSC::Handle<JSC::Unknown>, void* context) final;
};

inline JSC::WeakHandleOwner* wrapperOwner(DOMWrapperWorld&, TestDelegateToSharedSyntheticAttribute*)
{
    static NeverDestroyed<JSTestDelegateToSharedSyntheticAttributeOwner> owner;
    return &owner.get();
}

inline void* wrapperKey(TestDelegateToSharedSyntheticAttribute* wrappableObject)
{
    return wrappableObject;
}

JSC::JSValue toJS(JSC::JSGlobalObject*, JSDOMGlobalObject*, TestDelegateToSharedSyntheticAttribute&);
inline JSC::JSValue toJS(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, TestDelegateToSharedSyntheticAttribute* impl) { return impl ? toJS(lexicalGlobalObject, globalObject, *impl) : JSC::jsNull(); }
JSC::JSValue toJSNewlyCreated(JSC::JSGlobalObject*, JSDOMGlobalObject*, Ref<TestDelegateToSharedSyntheticAttribute>&&);
inline JSC::JSValue toJSNewlyCreated(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, RefPtr<TestDelegateToSharedSyntheticAttribute>&& impl) { return impl ? toJSNewlyCreated(lexicalGlobalObject, globalObject, impl.releaseNonNull()) : JSC::jsNull(); }

template<> struct JSDOMWrapperConverterTraits<TestDelegateToSharedSyntheticAttribute> {
    using WrapperClass = JSTestDelegateToSharedSyntheticAttribute;
    using ToWrappedReturnType = TestDelegateToSharedSyntheticAttribute*;
};

} // namespace WebCore
