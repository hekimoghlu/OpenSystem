/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 4, 2025.
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
#include "JSEventTarget.h"

namespace WebCore {

class WorkerGlobalScope;

class JSWorkerGlobalScope : public JSEventTarget {
public:
    using Base = JSEventTarget;
    using DOMWrapped = WorkerGlobalScope;
    static JSWorkerGlobalScope* create(JSC::VM& vm, JSC::Structure* structure, Ref<WorkerGlobalScope>&& impl, JSC::JSGlobalProxy* proxy)
    {
        JSWorkerGlobalScope* ptr = new (NotNull, JSC::allocateCell<JSWorkerGlobalScope>(vm)) JSWorkerGlobalScope(vm, structure, WTFMove(impl));
        ptr->finishCreation(vm, proxy);
        return ptr;
    }

    static JSC::JSObject* createPrototype(JSC::VM&, JSDOMGlobalObject&);
    static JSC::JSObject* prototype(JSC::VM&, JSDOMGlobalObject&);
    static WorkerGlobalScope* toWrapped(JSC::VM&, JSC::JSValue);

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
    WorkerGlobalScope& wrapped() const
    {
        return static_cast<WorkerGlobalScope&>(Base::wrapped());
    }

    Ref<WorkerGlobalScope> protectedWrapped() const;

public:
    static constexpr unsigned StructureFlags = Base::StructureFlags | JSC::HasStaticPropertyTable;
protected:
    JSWorkerGlobalScope(JSC::VM&, JSC::Structure*, Ref<WorkerGlobalScope>&&);
#if ASSERT_ENABLED
    void finishCreation(JSC::VM&, JSC::JSGlobalProxy*);
#endif
};


class JSWorkerGlobalScopePrototype final : public JSC::JSNonFinalObject {
public:
    using Base = JSC::JSNonFinalObject;
    static JSWorkerGlobalScopePrototype* create(JSC::VM& vm, JSDOMGlobalObject* globalObject, JSC::Structure* structure)
    {
        JSWorkerGlobalScopePrototype* ptr = new (NotNull, JSC::allocateCell<JSWorkerGlobalScopePrototype>(vm)) JSWorkerGlobalScopePrototype(vm, globalObject, structure);
        ptr->finishCreation(vm);
        return ptr;
    }

    DECLARE_INFO;
    template<typename CellType, JSC::SubspaceAccess>
    static JSC::GCClient::IsoSubspace* subspaceFor(JSC::VM& vm)
    {
        STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSWorkerGlobalScopePrototype, Base);
        return &vm.plainObjectSpace();
    }
    static JSC::Structure* createStructure(JSC::VM& vm, JSC::JSGlobalObject* globalObject, JSC::JSValue prototype)
    {
        return JSC::Structure::create(vm, globalObject, prototype, JSC::TypeInfo(JSC::ObjectType, StructureFlags), info());
    }

private:
    JSWorkerGlobalScopePrototype(JSC::VM& vm, JSC::JSGlobalObject*, JSC::Structure* structure)
        : JSC::JSNonFinalObject(vm, structure)
    {
    }

    void finishCreation(JSC::VM&);
public:
    static constexpr unsigned StructureFlags = Base::StructureFlags | JSC::HasStaticPropertyTable;
};
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSWorkerGlobalScopePrototype, JSWorkerGlobalScopePrototype::Base);

template<> struct JSDOMWrapperConverterTraits<WorkerGlobalScope> {
    using WrapperClass = JSWorkerGlobalScope;
    using ToWrappedReturnType = WorkerGlobalScope*;
};

} // namespace WebCore
