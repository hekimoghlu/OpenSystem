/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 15, 2023.
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
#include "TestTaggedWrapper.h"
#include <wtf/NeverDestroyed.h>

namespace WebCore {

WTF_DECLARE_PTRTAG(TestTaggedWrapperPtrTag)
class JSTestTaggedWrapper : public JSDOMWrapper<TestTaggedWrapper, SignedPtrTraits<TestTaggedWrapper, TestTaggedWrapperPtrTag>> {
public:
    using Base = JSDOMWrapper<TestTaggedWrapper, SignedPtrTraits<TestTaggedWrapper, TestTaggedWrapperPtrTag>>;
    static JSTestTaggedWrapper* create(JSC::Structure* structure, JSDOMGlobalObject* globalObject, Ref<TestTaggedWrapper>&& impl)
    {
        SUPPRESS_UNCOUNTED_LOCAL auto& vm = globalObject->vm();
        JSTestTaggedWrapper* ptr = new (NotNull, JSC::allocateCell<JSTestTaggedWrapper>(vm)) JSTestTaggedWrapper(structure, *globalObject, WTFMove(impl));
        ptr->finishCreation(vm);
        return ptr;
    }

    static JSC::JSObject* createPrototype(JSC::VM&, JSDOMGlobalObject&);
    static JSC::JSObject* prototype(JSC::VM&, JSDOMGlobalObject&);
    static TestTaggedWrapper* toWrapped(JSC::VM&, JSC::JSValue);
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
    JSTestTaggedWrapper(JSC::Structure*, JSDOMGlobalObject&, Ref<TestTaggedWrapper>&&);

    DECLARE_DEFAULT_FINISH_CREATION;
};

class JSTestTaggedWrapperOwner final : public JSC::WeakHandleOwner {
public:
    bool isReachableFromOpaqueRoots(JSC::Handle<JSC::Unknown>, void* context, JSC::AbstractSlotVisitor&, ASCIILiteral*) final;
    void finalize(JSC::Handle<JSC::Unknown>, void* context) final;
};

inline JSC::WeakHandleOwner* wrapperOwner(DOMWrapperWorld&, TestTaggedWrapper*)
{
    static NeverDestroyed<JSTestTaggedWrapperOwner> owner;
    return &owner.get();
}

inline void* wrapperKey(TestTaggedWrapper* wrappableObject)
{
    return wrappableObject;
}

JSC::JSValue toJS(JSC::JSGlobalObject*, JSDOMGlobalObject*, TestTaggedWrapper&);
inline JSC::JSValue toJS(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, TestTaggedWrapper* impl) { return impl ? toJS(lexicalGlobalObject, globalObject, *impl) : JSC::jsNull(); }
JSC::JSValue toJSNewlyCreated(JSC::JSGlobalObject*, JSDOMGlobalObject*, Ref<TestTaggedWrapper>&&);
inline JSC::JSValue toJSNewlyCreated(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, RefPtr<TestTaggedWrapper>&& impl) { return impl ? toJSNewlyCreated(lexicalGlobalObject, globalObject, impl.releaseNonNull()) : JSC::jsNull(); }

template<> struct JSDOMWrapperConverterTraits<TestTaggedWrapper> {
    using WrapperClass = JSTestTaggedWrapper;
    using ToWrappedReturnType = TestTaggedWrapper*;
};

} // namespace WebCore
