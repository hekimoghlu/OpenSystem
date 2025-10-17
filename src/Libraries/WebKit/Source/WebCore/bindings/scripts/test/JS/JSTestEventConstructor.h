/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 23, 2025.
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
#include "TestEventConstructor.h"

namespace WebCore {

class JSTestEventConstructor : public JSEvent {
public:
    using Base = JSEvent;
    using DOMWrapped = TestEventConstructor;
    static JSTestEventConstructor* create(JSC::Structure* structure, JSDOMGlobalObject* globalObject, Ref<TestEventConstructor>&& impl)
    {
        SUPPRESS_UNCOUNTED_LOCAL auto& vm = globalObject->vm();
        JSTestEventConstructor* ptr = new (NotNull, JSC::allocateCell<JSTestEventConstructor>(vm)) JSTestEventConstructor(structure, *globalObject, WTFMove(impl));
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
    TestEventConstructor& wrapped() const
    {
        return static_cast<TestEventConstructor&>(Base::wrapped());
    }

    Ref<TestEventConstructor> protectedWrapped() const;

protected:
    JSTestEventConstructor(JSC::Structure*, JSDOMGlobalObject&, Ref<TestEventConstructor>&&);

    DECLARE_DEFAULT_FINISH_CREATION;
};

JSC::JSValue toJS(JSC::JSGlobalObject*, JSDOMGlobalObject*, TestEventConstructor&);
inline JSC::JSValue toJS(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, TestEventConstructor* impl) { return impl ? toJS(lexicalGlobalObject, globalObject, *impl) : JSC::jsNull(); }
JSC::JSValue toJSNewlyCreated(JSC::JSGlobalObject*, JSDOMGlobalObject*, Ref<TestEventConstructor>&&);
inline JSC::JSValue toJSNewlyCreated(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, RefPtr<TestEventConstructor>&& impl) { return impl ? toJSNewlyCreated(lexicalGlobalObject, globalObject, impl.releaseNonNull()) : JSC::jsNull(); }

template<> struct JSDOMWrapperConverterTraits<TestEventConstructor> {
    using WrapperClass = JSTestEventConstructor;
    using ToWrappedReturnType = TestEventConstructor*;
};
template<> ConversionResult<IDLDictionary<TestEventConstructor::Init>> convertDictionary<TestEventConstructor::Init>(JSC::JSGlobalObject&, JSC::JSValue);


} // namespace WebCore
