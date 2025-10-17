/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 30, 2023.
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
#include "JSNode.h"
#include "TestDOMJIT.h"
#include <JavaScriptCore/DOMJITGetterSetter.h>

namespace WebCore {

class JSTestDOMJIT : public JSNode {
public:
    using Base = JSNode;
    using DOMWrapped = TestDOMJIT;
    static JSTestDOMJIT* create(JSC::Structure* structure, JSDOMGlobalObject* globalObject, Ref<TestDOMJIT>&& impl)
    {
        SUPPRESS_UNCOUNTED_LOCAL auto& vm = globalObject->vm();
        JSTestDOMJIT* ptr = new (NotNull, JSC::allocateCell<JSTestDOMJIT>(vm)) JSTestDOMJIT(structure, *globalObject, WTFMove(impl));
        ptr->finishCreation(vm);
        return ptr;
    }

    static JSC::JSObject* createPrototype(JSC::VM&, JSDOMGlobalObject&);
    static JSC::JSObject* prototype(JSC::VM&, JSDOMGlobalObject&);

    DECLARE_INFO;

    static JSC::Structure* createStructure(JSC::VM& vm, JSC::JSGlobalObject* globalObject, JSC::JSValue prototype)
    {
        return JSC::Structure::create(vm, globalObject, prototype, JSC::TypeInfo(JSC::JSType(JSNodeType), StructureFlags), info(), JSC::NonArray);
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
    TestDOMJIT& wrapped() const
    {
        return static_cast<TestDOMJIT&>(Base::wrapped());
    }

    Ref<TestDOMJIT> protectedWrapped() const;

protected:
    JSTestDOMJIT(JSC::Structure*, JSDOMGlobalObject&, Ref<TestDOMJIT>&&);

    DECLARE_DEFAULT_FINISH_CREATION;
};


// DOM JIT Attributes

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITAnyAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITBooleanAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITByteAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITOctetAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITShortAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITUnsignedShortAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITLongAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITUnsignedLongAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITLongLongAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITUnsignedLongLongAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITFloatAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITUnrestrictedFloatAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITDoubleAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITUnrestrictedDoubleAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITDomStringAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITByteStringAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITUsvStringAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITNodeAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITBooleanNullableAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITByteNullableAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITOctetNullableAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITShortNullableAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITUnsignedShortNullableAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITLongNullableAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITUnsignedLongNullableAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITLongLongNullableAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITUnsignedLongLongNullableAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITFloatNullableAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITUnrestrictedFloatNullableAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITDoubleNullableAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITUnrestrictedDoubleNullableAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITDomStringNullableAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITByteStringNullableAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITUsvStringNullableAttrAttribute();
#endif

#if ENABLE(JIT)
Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileTestDOMJITNodeNullableAttrAttribute();
#endif

template<> struct JSDOMWrapperConverterTraits<TestDOMJIT> {
    using WrapperClass = JSTestDOMJIT;
    using ToWrappedReturnType = TestDOMJIT*;
};

} // namespace WebCore
