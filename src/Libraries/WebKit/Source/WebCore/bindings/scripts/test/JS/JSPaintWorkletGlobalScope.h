/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 18, 2024.
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
#include "JSWorkletGlobalScope.h"

namespace WebCore {

class PaintWorkletGlobalScope;

class JSPaintWorkletGlobalScope : public JSWorkletGlobalScope {
public:
    using Base = JSWorkletGlobalScope;
    using DOMWrapped = PaintWorkletGlobalScope;
    static JSPaintWorkletGlobalScope* create(JSC::VM& vm, JSC::Structure* structure, Ref<PaintWorkletGlobalScope>&& impl, JSC::JSGlobalProxy* proxy)
    {
        JSPaintWorkletGlobalScope* ptr = new (NotNull, JSC::allocateCell<JSPaintWorkletGlobalScope>(vm)) JSPaintWorkletGlobalScope(vm, structure, WTFMove(impl));
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
    PaintWorkletGlobalScope& wrapped() const
    {
        return static_cast<PaintWorkletGlobalScope&>(Base::wrapped());
    }

    Ref<PaintWorkletGlobalScope> protectedWrapped() const;

public:
    static constexpr unsigned StructureFlags = Base::StructureFlags | JSC::HasStaticPropertyTable;
protected:
    JSPaintWorkletGlobalScope(JSC::VM&, JSC::Structure*, Ref<PaintWorkletGlobalScope>&&);
#if ASSERT_ENABLED
    void finishCreation(JSC::VM&, JSC::JSGlobalProxy*);
#endif
};


class JSPaintWorkletGlobalScopePrototype final : public JSC::JSNonFinalObject {
public:
    using Base = JSC::JSNonFinalObject;
    static JSPaintWorkletGlobalScopePrototype* create(JSC::VM& vm, JSDOMGlobalObject* globalObject, JSC::Structure* structure)
    {
        JSPaintWorkletGlobalScopePrototype* ptr = new (NotNull, JSC::allocateCell<JSPaintWorkletGlobalScopePrototype>(vm)) JSPaintWorkletGlobalScopePrototype(vm, globalObject, structure);
        ptr->finishCreation(vm);
        return ptr;
    }

    DECLARE_INFO;
    template<typename CellType, JSC::SubspaceAccess>
    static JSC::GCClient::IsoSubspace* subspaceFor(JSC::VM& vm)
    {
        STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSPaintWorkletGlobalScopePrototype, Base);
        return &vm.plainObjectSpace();
    }
    static JSC::Structure* createStructure(JSC::VM& vm, JSC::JSGlobalObject* globalObject, JSC::JSValue prototype)
    {
        return JSC::Structure::create(vm, globalObject, prototype, JSC::TypeInfo(JSC::ObjectType, StructureFlags), info());
    }

private:
    JSPaintWorkletGlobalScopePrototype(JSC::VM& vm, JSC::JSGlobalObject*, JSC::Structure* structure)
        : JSC::JSNonFinalObject(vm, structure)
    {
    }

    void finishCreation(JSC::VM&);
public:
    static constexpr unsigned StructureFlags = Base::StructureFlags | JSC::HasStaticPropertyTable;
};
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSPaintWorkletGlobalScopePrototype, JSPaintWorkletGlobalScopePrototype::Base);

template<> struct JSDOMWrapperConverterTraits<PaintWorkletGlobalScope> {
    using WrapperClass = JSPaintWorkletGlobalScope;
    using ToWrappedReturnType = PaintWorkletGlobalScope*;
};

} // namespace WebCore
