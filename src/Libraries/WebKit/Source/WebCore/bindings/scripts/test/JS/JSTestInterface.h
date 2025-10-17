/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 2, 2025.
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

#if ENABLE(Condition1) || ENABLE(Condition2)

#include "JSDOMWrapper.h"
#include "TestInterface.h"
#include <wtf/NeverDestroyed.h>

namespace WebCore {

class WEBCORE_EXPORT JSTestInterface : public JSDOMWrapper<TestInterface> {
public:
    using Base = JSDOMWrapper<TestInterface>;
    static JSTestInterface* create(JSC::Structure* structure, JSDOMGlobalObject* globalObject, Ref<TestInterface>&& impl)
    {
        SUPPRESS_UNCOUNTED_LOCAL auto& vm = globalObject->vm();
        JSTestInterface* ptr = new (NotNull, JSC::allocateCell<JSTestInterface>(vm)) JSTestInterface(structure, *globalObject, WTFMove(impl));
        ptr->finishCreation(vm);
        return ptr;
    }

    static JSC::JSObject* createPrototype(JSC::VM&, JSDOMGlobalObject&);
    static JSC::JSObject* prototype(JSC::VM&, JSDOMGlobalObject&);
    static TestInterface* toWrapped(JSC::VM&, JSC::JSValue);
    static bool put(JSC::JSCell*, JSC::JSGlobalObject*, JSC::PropertyName, JSC::JSValue, JSC::PutPropertySlot&);
    static bool putByIndex(JSC::JSCell*, JSC::JSGlobalObject*, unsigned propertyName, JSC::JSValue, bool shouldThrow);
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

    // Custom attributes
#if ENABLE(Condition22) || ENABLE(Condition23)
    JSC::JSValue mixinCustomAttribute(JSC::JSGlobalObject&) const;
#endif
#if ENABLE(Condition22) || ENABLE(Condition23)
    void setMixinCustomAttribute(JSC::JSGlobalObject&, JSC::JSValue);
#endif
#if ENABLE(Condition11) || ENABLE(Condition12)
    JSC::JSValue supplementalStr3(JSC::JSGlobalObject&) const;
#endif
#if ENABLE(Condition11) || ENABLE(Condition12)
    void setSupplementalStr3(JSC::JSGlobalObject&, JSC::JSValue);
#endif

    // Custom functions
#if ENABLE(Condition22) || ENABLE(Condition23)
    JSC::JSValue mixinCustomOperation(JSC::JSGlobalObject&, JSC::CallFrame&);
#endif
#if ENABLE(Condition11) || ENABLE(Condition12)
    JSC::JSValue supplementalMethod3(JSC::JSGlobalObject&, JSC::CallFrame&);
#endif
public:
    static constexpr unsigned StructureFlags = Base::StructureFlags | JSC::OverridesPut;
protected:
    JSTestInterface(JSC::Structure*, JSDOMGlobalObject&, Ref<TestInterface>&&);

    DECLARE_DEFAULT_FINISH_CREATION;
};

class WEBCORE_EXPORT JSTestInterfaceOwner final : public JSC::WeakHandleOwner {
public:
    bool isReachableFromOpaqueRoots(JSC::Handle<JSC::Unknown>, void* context, JSC::AbstractSlotVisitor&, ASCIILiteral*) final;
    void finalize(JSC::Handle<JSC::Unknown>, void* context) final;
};

inline JSC::WeakHandleOwner* wrapperOwner(DOMWrapperWorld&, TestInterface*)
{
    static NeverDestroyed<JSTestInterfaceOwner> owner;
    return &owner.get();
}

inline void* wrapperKey(TestInterface* wrappableObject)
{
    return wrappableObject;
}

WEBCORE_EXPORT JSC::JSValue toJS(JSC::JSGlobalObject*, JSDOMGlobalObject*, TestInterface&);
inline JSC::JSValue toJS(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, TestInterface* impl) { return impl ? toJS(lexicalGlobalObject, globalObject, *impl) : JSC::jsNull(); }
JSC::JSValue toJSNewlyCreated(JSC::JSGlobalObject*, JSDOMGlobalObject*, Ref<TestInterface>&&);
inline JSC::JSValue toJSNewlyCreated(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, RefPtr<TestInterface>&& impl) { return impl ? toJSNewlyCreated(lexicalGlobalObject, globalObject, impl.releaseNonNull()) : JSC::jsNull(); }

template<> struct JSDOMWrapperConverterTraits<TestInterface> {
    using WrapperClass = JSTestInterface;
    using ToWrappedReturnType = TestInterface*;
};

} // namespace WebCore

#endif // ENABLE(Condition1) || ENABLE(Condition2)
