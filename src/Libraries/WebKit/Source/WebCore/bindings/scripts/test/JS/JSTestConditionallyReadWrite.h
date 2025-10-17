/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 14, 2023.
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
#include "TestConditionallyReadWrite.h"
#include <wtf/NeverDestroyed.h>

namespace WebCore {

class JSTestConditionallyReadWrite : public JSDOMWrapper<TestConditionallyReadWrite> {
public:
    using Base = JSDOMWrapper<TestConditionallyReadWrite>;
    static JSTestConditionallyReadWrite* create(JSC::Structure* structure, JSDOMGlobalObject* globalObject, Ref<TestConditionallyReadWrite>&& impl)
    {
        SUPPRESS_UNCOUNTED_LOCAL auto& vm = globalObject->vm();
        JSTestConditionallyReadWrite* ptr = new (NotNull, JSC::allocateCell<JSTestConditionallyReadWrite>(vm)) JSTestConditionallyReadWrite(structure, *globalObject, WTFMove(impl));
        ptr->finishCreation(vm);
        return ptr;
    }

    static JSC::JSObject* createPrototype(JSC::VM&, JSDOMGlobalObject&);
    static JSC::JSObject* prototype(JSC::VM&, JSDOMGlobalObject&);
    static TestConditionallyReadWrite* toWrapped(JSC::VM&, JSC::JSValue);
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
public:
    static constexpr unsigned StructureFlags = Base::StructureFlags | JSC::HasStaticPropertyTable;
protected:
    JSTestConditionallyReadWrite(JSC::Structure*, JSDOMGlobalObject&, Ref<TestConditionallyReadWrite>&&);

    void finishCreation(JSC::VM&);
};

class JSTestConditionallyReadWriteOwner final : public JSC::WeakHandleOwner {
public:
    bool isReachableFromOpaqueRoots(JSC::Handle<JSC::Unknown>, void* context, JSC::AbstractSlotVisitor&, ASCIILiteral*) final;
    void finalize(JSC::Handle<JSC::Unknown>, void* context) final;
};

inline JSC::WeakHandleOwner* wrapperOwner(DOMWrapperWorld&, TestConditionallyReadWrite*)
{
    static NeverDestroyed<JSTestConditionallyReadWriteOwner> owner;
    return &owner.get();
}

inline void* wrapperKey(TestConditionallyReadWrite* wrappableObject)
{
    return wrappableObject;
}

JSC::JSValue toJS(JSC::JSGlobalObject*, JSDOMGlobalObject*, TestConditionallyReadWrite&);
inline JSC::JSValue toJS(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, TestConditionallyReadWrite* impl) { return impl ? toJS(lexicalGlobalObject, globalObject, *impl) : JSC::jsNull(); }
JSC::JSValue toJSNewlyCreated(JSC::JSGlobalObject*, JSDOMGlobalObject*, Ref<TestConditionallyReadWrite>&&);
inline JSC::JSValue toJSNewlyCreated(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, RefPtr<TestConditionallyReadWrite>&& impl) { return impl ? toJSNewlyCreated(lexicalGlobalObject, globalObject, impl.releaseNonNull()) : JSC::jsNull(); }

template<> struct JSDOMWrapperConverterTraits<TestConditionallyReadWrite> {
    using WrapperClass = JSTestConditionallyReadWrite;
    using ToWrappedReturnType = TestConditionallyReadWrite*;
};

} // namespace WebCore
