/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 19, 2023.
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
#include "JSWorkerGlobalScope.h"

namespace WebCore {

class SharedWorkerGlobalScope;

class JSSharedWorkerGlobalScope : public JSWorkerGlobalScope {
public:
    using Base = JSWorkerGlobalScope;
    using DOMWrapped = SharedWorkerGlobalScope;
    static JSSharedWorkerGlobalScope* create(JSC::VM& vm, JSC::Structure* structure, Ref<SharedWorkerGlobalScope>&& impl, JSC::JSGlobalProxy* proxy)
    {
        JSSharedWorkerGlobalScope* ptr = new (NotNull, JSC::allocateCell<JSSharedWorkerGlobalScope>(vm)) JSSharedWorkerGlobalScope(vm, structure, WTFMove(impl));
        ptr->finishCreation(vm, proxy);
        return ptr;
    }


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
    SharedWorkerGlobalScope& wrapped() const
    {
        return static_cast<SharedWorkerGlobalScope&>(Base::wrapped());
    }

    Ref<SharedWorkerGlobalScope> protectedWrapped() const;

public:
    static constexpr unsigned StructureFlags = Base::StructureFlags | JSC::HasStaticPropertyTable;
protected:
    JSSharedWorkerGlobalScope(JSC::VM&, JSC::Structure*, Ref<SharedWorkerGlobalScope>&&);
#if ASSERT_ENABLED
    void finishCreation(JSC::VM&, JSC::JSGlobalProxy*);
#endif
};


class JSSharedWorkerGlobalScopePrototype final : public JSC::JSNonFinalObject {
public:
    using Base = JSC::JSNonFinalObject;
    static JSSharedWorkerGlobalScopePrototype* create(JSC::VM& vm, JSDOMGlobalObject* globalObject, JSC::Structure* structure)
    {
        JSSharedWorkerGlobalScopePrototype* ptr = new (NotNull, JSC::allocateCell<JSSharedWorkerGlobalScopePrototype>(vm)) JSSharedWorkerGlobalScopePrototype(vm, globalObject, structure);
        ptr->finishCreation(vm);
        return ptr;
    }

    DECLARE_INFO;
    template<typename CellType, JSC::SubspaceAccess>
    static JSC::GCClient::IsoSubspace* subspaceFor(JSC::VM& vm)
    {
        STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSSharedWorkerGlobalScopePrototype, Base);
        return &vm.plainObjectSpace();
    }
    static JSC::Structure* createStructure(JSC::VM& vm, JSC::JSGlobalObject* globalObject, JSC::JSValue prototype)
    {
        return JSC::Structure::create(vm, globalObject, prototype, JSC::TypeInfo(JSC::ObjectType, StructureFlags), info());
    }

private:
    JSSharedWorkerGlobalScopePrototype(JSC::VM& vm, JSC::JSGlobalObject*, JSC::Structure* structure)
        : JSC::JSNonFinalObject(vm, structure)
    {
    }

    void finishCreation(JSC::VM&);
public:
    static constexpr unsigned StructureFlags = Base::StructureFlags | JSC::HasStaticPropertyTable;
};
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSSharedWorkerGlobalScopePrototype, JSSharedWorkerGlobalScopePrototype::Base);

template<> struct JSDOMWrapperConverterTraits<SharedWorkerGlobalScope> {
    using WrapperClass = JSSharedWorkerGlobalScope;
    using ToWrappedReturnType = SharedWorkerGlobalScope*;
};

} // namespace WebCore
