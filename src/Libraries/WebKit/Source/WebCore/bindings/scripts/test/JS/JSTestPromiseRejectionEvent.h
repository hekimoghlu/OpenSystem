/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 2, 2025.
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

#include "JSDOMConvertDictionary.h"
#include "JSDOMWrapper.h"
#include "JSEvent.h"
#include "TestPromiseRejectionEvent.h"

namespace WebCore {

class JSTestPromiseRejectionEvent : public JSEvent {
public:
    using Base = JSEvent;
    using DOMWrapped = TestPromiseRejectionEvent;
    static JSTestPromiseRejectionEvent* create(JSC::Structure* structure, JSDOMGlobalObject* globalObject, Ref<TestPromiseRejectionEvent>&& impl)
    {
        SUPPRESS_UNCOUNTED_LOCAL auto& vm = globalObject->vm();
        JSTestPromiseRejectionEvent* ptr = new (NotNull, JSC::allocateCell<JSTestPromiseRejectionEvent>(vm)) JSTestPromiseRejectionEvent(structure, *globalObject, WTFMove(impl));
        ptr->finishCreation(vm);
        return ptr;
    }

    static JSC::JSObject* createPrototype(JSC::VM&, JSDOMGlobalObject&);
    static JSC::JSObject* prototype(JSC::VM&, JSDOMGlobalObject&);

    DECLARE_INFO;

    static JSC::Structure* createStructure(JSC::VM& vm, JSC::JSGlobalObject* globalObject, JSC::JSValue prototype)
    {
        return JSC::Structure::create(vm, globalObject, prototype, JSC::TypeInfo(JSC::JSType(JSEventType), StructureFlags), info(), JSC::NonArray);
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
    TestPromiseRejectionEvent& wrapped() const
    {
        return static_cast<TestPromiseRejectionEvent&>(Base::wrapped());
    }

    Ref<TestPromiseRejectionEvent> protectedWrapped() const;

protected:
    JSTestPromiseRejectionEvent(JSC::Structure*, JSDOMGlobalObject&, Ref<TestPromiseRejectionEvent>&&);

    DECLARE_DEFAULT_FINISH_CREATION;
};

JSC::JSValue toJS(JSC::JSGlobalObject*, JSDOMGlobalObject*, TestPromiseRejectionEvent&);
inline JSC::JSValue toJS(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, TestPromiseRejectionEvent* impl) { return impl ? toJS(lexicalGlobalObject, globalObject, *impl) : JSC::jsNull(); }
JSC::JSValue toJSNewlyCreated(JSC::JSGlobalObject*, JSDOMGlobalObject*, Ref<TestPromiseRejectionEvent>&&);
inline JSC::JSValue toJSNewlyCreated(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, RefPtr<TestPromiseRejectionEvent>&& impl) { return impl ? toJSNewlyCreated(lexicalGlobalObject, globalObject, impl.releaseNonNull()) : JSC::jsNull(); }

template<> struct JSDOMWrapperConverterTraits<TestPromiseRejectionEvent> {
    using WrapperClass = JSTestPromiseRejectionEvent;
    using ToWrappedReturnType = TestPromiseRejectionEvent*;
};
template<> ConversionResult<IDLDictionary<TestPromiseRejectionEvent::Init>> convertDictionary<TestPromiseRejectionEvent::Init>(JSC::JSGlobalObject&, JSC::JSValue);


} // namespace WebCore
