/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 25, 2023.
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
#include "JSTestDefaultToJSONInherit.h"
#include "TestDefaultToJSONIndirectInheritance.h"

namespace WebCore {

class JSTestDefaultToJSONIndirectInheritance : public JSTestDefaultToJSONInherit {
public:
    using Base = JSTestDefaultToJSONInherit;
    using DOMWrapped = TestDefaultToJSONIndirectInheritance;
    static JSTestDefaultToJSONIndirectInheritance* create(JSC::Structure* structure, JSDOMGlobalObject* globalObject, Ref<TestDefaultToJSONIndirectInheritance>&& impl)
    {
        SUPPRESS_UNCOUNTED_LOCAL auto& vm = globalObject->vm();
        JSTestDefaultToJSONIndirectInheritance* ptr = new (NotNull, JSC::allocateCell<JSTestDefaultToJSONIndirectInheritance>(vm)) JSTestDefaultToJSONIndirectInheritance(structure, *globalObject, WTFMove(impl));
        ptr->finishCreation(vm);
        return ptr;
    }

    static JSC::JSObject* createPrototype(JSC::VM&, JSDOMGlobalObject&);
    static JSC::JSObject* prototype(JSC::VM&, JSDOMGlobalObject&);

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
    TestDefaultToJSONIndirectInheritance& wrapped() const
    {
        return static_cast<TestDefaultToJSONIndirectInheritance&>(Base::wrapped());
    }

    Ref<TestDefaultToJSONIndirectInheritance> protectedWrapped() const;

protected:
    JSTestDefaultToJSONIndirectInheritance(JSC::Structure*, JSDOMGlobalObject&, Ref<TestDefaultToJSONIndirectInheritance>&&);

    DECLARE_DEFAULT_FINISH_CREATION;
};


template<> struct JSDOMWrapperConverterTraits<TestDefaultToJSONIndirectInheritance> {
    using WrapperClass = JSTestDefaultToJSONIndirectInheritance;
    using ToWrappedReturnType = TestDefaultToJSONIndirectInheritance*;
};

} // namespace WebCore
