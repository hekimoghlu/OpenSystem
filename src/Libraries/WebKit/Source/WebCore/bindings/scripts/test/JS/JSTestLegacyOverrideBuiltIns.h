/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 3, 2022.
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
#include "TestLegacyOverrideBuiltIns.h"
#include <wtf/NeverDestroyed.h>

namespace WebCore {

class JSTestLegacyOverrideBuiltIns : public JSDOMWrapper<TestLegacyOverrideBuiltIns> {
public:
    using Base = JSDOMWrapper<TestLegacyOverrideBuiltIns>;
    static JSTestLegacyOverrideBuiltIns* create(JSC::Structure* structure, JSDOMGlobalObject* globalObject, Ref<TestLegacyOverrideBuiltIns>&& impl)
    {
        SUPPRESS_UNCOUNTED_LOCAL auto& vm = globalObject->vm();
        JSTestLegacyOverrideBuiltIns* ptr = new (NotNull, JSC::allocateCell<JSTestLegacyOverrideBuiltIns>(vm)) JSTestLegacyOverrideBuiltIns(structure, *globalObject, WTFMove(impl));
        ptr->finishCreation(vm);
        return ptr;
    }

    static JSC::JSObject* createPrototype(JSC::VM&, JSDOMGlobalObject&);
    static JSC::JSObject* prototype(JSC::VM&, JSDOMGlobalObject&);
    static TestLegacyOverrideBuiltIns* toWrapped(JSC::VM&, JSC::JSValue);
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
    static constexpr unsigned StructureFlags = Base::StructureFlags | JSC::GetOwnPropertySlotIsImpure | JSC::InterceptsGetOwnPropertySlotByIndexEvenWhenLengthIsNotZero | JSC::OverridesGetOwnPropertyNames | JSC::OverridesGetOwnPropertySlot | JSC::OverridesPut;
protected:
    JSTestLegacyOverrideBuiltIns(JSC::Structure*, JSDOMGlobalObject&, Ref<TestLegacyOverrideBuiltIns>&&);

    DECLARE_DEFAULT_FINISH_CREATION;
};

class JSTestLegacyOverrideBuiltInsOwner final : public JSC::WeakHandleOwner {
public:
    bool isReachableFromOpaqueRoots(JSC::Handle<JSC::Unknown>, void* context, JSC::AbstractSlotVisitor&, ASCIILiteral*) final;
    void finalize(JSC::Handle<JSC::Unknown>, void* context) final;
};

inline JSC::WeakHandleOwner* wrapperOwner(DOMWrapperWorld&, TestLegacyOverrideBuiltIns*)
{
    static NeverDestroyed<JSTestLegacyOverrideBuiltInsOwner> owner;
    return &owner.get();
}

inline void* wrapperKey(TestLegacyOverrideBuiltIns* wrappableObject)
{
    return wrappableObject;
}

JSC::JSValue toJS(JSC::JSGlobalObject*, JSDOMGlobalObject*, TestLegacyOverrideBuiltIns&);
inline JSC::JSValue toJS(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, TestLegacyOverrideBuiltIns* impl) { return impl ? toJS(lexicalGlobalObject, globalObject, *impl) : JSC::jsNull(); }
JSC::JSValue toJSNewlyCreated(JSC::JSGlobalObject*, JSDOMGlobalObject*, Ref<TestLegacyOverrideBuiltIns>&&);
inline JSC::JSValue toJSNewlyCreated(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, RefPtr<TestLegacyOverrideBuiltIns>&& impl) { return impl ? toJSNewlyCreated(lexicalGlobalObject, globalObject, impl.releaseNonNull()) : JSC::jsNull(); }

template<> struct JSDOMWrapperConverterTraits<TestLegacyOverrideBuiltIns> {
    using WrapperClass = JSTestLegacyOverrideBuiltIns;
    using ToWrappedReturnType = TestLegacyOverrideBuiltIns*;
};

} // namespace WebCore
