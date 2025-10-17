/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 4, 2022.
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

#include "InterfaceName.h"
#include "JSDOMWrapper.h"
#include <wtf/NeverDestroyed.h>

namespace WebCore {

class JSInterfaceName : public JSDOMWrapper<InterfaceName> {
public:
    using Base = JSDOMWrapper<InterfaceName>;
    static JSInterfaceName* create(JSC::Structure* structure, JSDOMGlobalObject* globalObject, Ref<InterfaceName>&& impl)
    {
        JSInterfaceName* ptr = new (NotNull, JSC::allocateCell<JSInterfaceName>(globalObject->vm().heap)) JSInterfaceName(structure, *globalObject, WTFMove(impl));
        ptr->finishCreation(globalObject->vm());
        return ptr;
    }

    static JSC::JSObject* createPrototype(JSC::VM&, JSDOMGlobalObject&);
    static JSC::JSObject* prototype(JSC::VM&, JSDOMGlobalObject&);
    static InterfaceName* toWrapped(JSC::VM&, JSC::JSValue);
    static size_t estimatedSize(JSCell*, JSC::VM&);
    static void destroy(JSC::JSCell*);

    DECLARE_INFO;

    static JSC::Structure* createStructure(JSC::VM& vm, JSC::JSGlobalObject* globalObject, JSC::JSValue prototype)
    {
        return JSC::Structure::create(vm, globalObject, prototype, JSC::TypeInfo(JSC::ObjectType, StructureFlags), info(), JSC::NonArray);
    }

    static JSC::JSValue getConstructor(JSC::VM&, const JSC::JSGlobalObject*);
    template<typename, JSC::SubspaceAccess mode> static JSC::IsoSubspace* subspaceFor(JSC::VM& vm)
    {
        if constexpr (mode == JSC::SubspaceAccess::Concurrently)
            return nullptr;
        return subspaceForImpl(vm);
    }
    static JSC::IsoSubspace* subspaceForImpl(JSC::VM& vm);
    static void visitChildren(JSCell*, JSC::SlotVisitor&);

    static void analyzeHeap(JSCell*, JSC::HeapAnalyzer&);
protected:
    JSInterfaceName(JSC::Structure*, JSDOMGlobalObject&, Ref<InterfaceName>&&);

    void finishCreation(JSC::VM&);
};

class JSInterfaceNameOwner : public JSC::WeakHandleOwner {
public:
    virtual bool isReachableFromOpaqueRoots(JSC::Handle<JSC::Unknown>, void* context, JSC::SlotVisitor&, const char**);
    virtual void finalize(JSC::Handle<JSC::Unknown>, void* context);
};

inline JSC::WeakHandleOwner* wrapperOwner(DOMWrapperWorld&, InterfaceName*)
{
    static NeverDestroyed<JSInterfaceNameOwner> owner;
    return &owner.get();
}

inline void* wrapperKey(InterfaceName* wrappableObject)
{
    return wrappableObject;
}

JSC::JSValue toJS(JSC::JSGlobalObject*, JSDOMGlobalObject*, InterfaceName&);
inline JSC::JSValue toJS(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, InterfaceName* impl) { return impl ? toJS(lexicalGlobalObject, globalObject, *impl) : JSC::jsNull(); }
JSC::JSValue toJSNewlyCreated(JSC::JSGlobalObject*, JSDOMGlobalObject*, Ref<InterfaceName>&&);
inline JSC::JSValue toJSNewlyCreated(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, RefPtr<InterfaceName>&& impl) { return impl ? toJSNewlyCreated(lexicalGlobalObject, globalObject, impl.releaseNonNull()) : JSC::jsNull(); }

template<> struct JSDOMWrapperConverterTraits<InterfaceName> {
    using WrapperClass = JSInterfaceName;
    using ToWrappedReturnType = InterfaceName*;
};

} // namespace WebCore
