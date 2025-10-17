/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 20, 2024.
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
#include "TestIndexedSetterNoIdentifier.h"
#include <wtf/NeverDestroyed.h>

namespace WebCore {

class JSTestIndexedSetterNoIdentifier : public JSDOMWrapper<TestIndexedSetterNoIdentifier> {
public:
    using Base = JSDOMWrapper<TestIndexedSetterNoIdentifier>;
    static JSTestIndexedSetterNoIdentifier* create(JSC::Structure* structure, JSDOMGlobalObject* globalObject, Ref<TestIndexedSetterNoIdentifier>&& impl)
    {
        SUPPRESS_UNCOUNTED_LOCAL auto& vm = globalObject->vm();
        JSTestIndexedSetterNoIdentifier* ptr = new (NotNull, JSC::allocateCell<JSTestIndexedSetterNoIdentifier>(vm)) JSTestIndexedSetterNoIdentifier(structure, *globalObject, WTFMove(impl));
        ptr->finishCreation(vm);
        return ptr;
    }

    static JSC::JSObject* createPrototype(JSC::VM&, JSDOMGlobalObject&);
    static JSC::JSObject* prototype(JSC::VM&, JSDOMGlobalObject&);
    static TestIndexedSetterNoIdentifier* toWrapped(JSC::VM&, JSC::JSValue);
    static bool legacyPlatformObjectGetOwnProperty(JSC::JSObject*, JSC::JSGlobalObject*, JSC::PropertyName, JSC::PropertySlot&, bool ignoreNamedProperties);
    static bool getOwnPropertySlot(JSC::JSObject*, JSC::JSGlobalObject*, JSC::PropertyName, JSC::PropertySlot&);
    static bool getOwnPropertySlotByIndex(JSC::JSObject*, JSC::JSGlobalObject*, unsigned propertyName, JSC::PropertySlot&);
    static void getOwnPropertyNames(JSC::JSObject*, JSC::JSGlobalObject*, JSC::PropertyNameArray&, JSC::DontEnumPropertiesMode);
    static bool put(JSC::JSCell*, JSC::JSGlobalObject*, JSC::PropertyName, JSC::JSValue, JSC::PutPropertySlot&);
    static bool putByIndex(JSC::JSCell*, JSC::JSGlobalObject*, unsigned propertyName, JSC::JSValue, bool shouldThrow);
    static bool defineOwnProperty(JSC::JSObject*, JSC::JSGlobalObject*, JSC::PropertyName, const JSC::PropertyDescriptor&, bool shouldThrow);
    static bool deleteProperty(JSC::JSCell*, JSC::JSGlobalObject*, JSC::PropertyName, JSC::DeletePropertySlot&);
    static bool deletePropertyByIndex(JSC::JSCell*, JSC::JSGlobalObject*, unsigned);
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
    static void analyzeHeap(JSCell*, JSC::HeapAnalyzer&);
public:
    static constexpr unsigned StructureFlags = Base::StructureFlags | JSC::InterceptsGetOwnPropertySlotByIndexEvenWhenLengthIsNotZero | JSC::OverridesGetOwnPropertyNames | JSC::OverridesGetOwnPropertySlot | JSC::OverridesPut;
protected:
    JSTestIndexedSetterNoIdentifier(JSC::Structure*, JSDOMGlobalObject&, Ref<TestIndexedSetterNoIdentifier>&&);

    DECLARE_DEFAULT_FINISH_CREATION;
};

class JSTestIndexedSetterNoIdentifierOwner final : public JSC::WeakHandleOwner {
public:
    bool isReachableFromOpaqueRoots(JSC::Handle<JSC::Unknown>, void* context, JSC::AbstractSlotVisitor&, ASCIILiteral*) final;
    void finalize(JSC::Handle<JSC::Unknown>, void* context) final;
};

inline JSC::WeakHandleOwner* wrapperOwner(DOMWrapperWorld&, TestIndexedSetterNoIdentifier*)
{
    static NeverDestroyed<JSTestIndexedSetterNoIdentifierOwner> owner;
    return &owner.get();
}

inline void* wrapperKey(TestIndexedSetterNoIdentifier* wrappableObject)
{
    return wrappableObject;
}

JSC::JSValue toJS(JSC::JSGlobalObject*, JSDOMGlobalObject*, TestIndexedSetterNoIdentifier&);
inline JSC::JSValue toJS(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, TestIndexedSetterNoIdentifier* impl) { return impl ? toJS(lexicalGlobalObject, globalObject, *impl) : JSC::jsNull(); }
JSC::JSValue toJSNewlyCreated(JSC::JSGlobalObject*, JSDOMGlobalObject*, Ref<TestIndexedSetterNoIdentifier>&&);
inline JSC::JSValue toJSNewlyCreated(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, RefPtr<TestIndexedSetterNoIdentifier>&& impl) { return impl ? toJSNewlyCreated(lexicalGlobalObject, globalObject, impl.releaseNonNull()) : JSC::jsNull(); }

template<> struct JSDOMWrapperConverterTraits<TestIndexedSetterNoIdentifier> {
    using WrapperClass = JSTestIndexedSetterNoIdentifier;
    using ToWrappedReturnType = TestIndexedSetterNoIdentifier*;
};

} // namespace WebCore
