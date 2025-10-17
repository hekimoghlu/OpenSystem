/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 19, 2021.
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
#include <wtf/NeverDestroyed.h>

namespace WebCore {

class ShadowRealmGlobalScope;

class JSShadowRealmGlobalScope : public JSDOMWrapper<ShadowRealmGlobalScope> {
public:
    using Base = JSDOMWrapper<ShadowRealmGlobalScope>;
    static JSShadowRealmGlobalScope* create(JSC::VM& vm, JSC::Structure* structure, Ref<ShadowRealmGlobalScope>&& impl, JSC::JSGlobalProxy* proxy)
    {
        JSShadowRealmGlobalScope* ptr = new (NotNull, JSC::allocateCell<JSShadowRealmGlobalScope>(vm)) JSShadowRealmGlobalScope(vm, structure, WTFMove(impl));
        ptr->finishCreation(vm, proxy);
        return ptr;
    }

    static ShadowRealmGlobalScope* toWrapped(JSC::VM&, JSC::JSValue);
    static void destroy(JSC::JSCell*);

    DECLARE_INFO;

    static JSC::Structure* createStructure(JSC::VM& vm, JSC::JSGlobalObject* globalObject, JSC::JSValue prototype)
    {
        return JSC::Structure::create(vm, globalObject, prototype, JSC::TypeInfo(JSC::GlobalObjectType, StructureFlags), info(), JSC::NonArray);
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
    static constexpr unsigned StructureFlags = (Base::StructureFlags & ~JSC::IsImmutablePrototypeExoticObject) | JSC::HasStaticPropertyTable;
protected:
    JSShadowRealmGlobalScope(JSC::VM&, JSC::Structure*, Ref<ShadowRealmGlobalScope>&&);
    void finishCreation(JSC::VM&, JSC::JSGlobalProxy*);
};

class JSShadowRealmGlobalScopeOwner final : public JSC::WeakHandleOwner {
public:
    bool isReachableFromOpaqueRoots(JSC::Handle<JSC::Unknown>, void* context, JSC::AbstractSlotVisitor&, ASCIILiteral*) final;
    void finalize(JSC::Handle<JSC::Unknown>, void* context) final;
};

inline JSC::WeakHandleOwner* wrapperOwner(DOMWrapperWorld&, ShadowRealmGlobalScope*)
{
    static NeverDestroyed<JSShadowRealmGlobalScopeOwner> owner;
    return &owner.get();
}

inline void* wrapperKey(ShadowRealmGlobalScope* wrappableObject)
{
    return wrappableObject;
}


template<> struct JSDOMWrapperConverterTraits<ShadowRealmGlobalScope> {
    using WrapperClass = JSShadowRealmGlobalScope;
    using ToWrappedReturnType = ShadowRealmGlobalScope*;
};

} // namespace WebCore
