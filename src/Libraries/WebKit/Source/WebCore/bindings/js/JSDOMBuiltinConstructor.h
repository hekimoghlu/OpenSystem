/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 6, 2023.
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

#include "JSDOMBuiltinConstructorBase.h"
#include "JSDOMExceptionHandling.h"
#include "JSDOMWrapperCache.h"

namespace WebCore {

template<typename JSClass> class JSDOMBuiltinConstructor final : public JSDOMBuiltinConstructorBase {
public:
    using Base = JSDOMBuiltinConstructorBase;

    static JSDOMBuiltinConstructor* create(JSC::VM&, JSC::Structure*, JSDOMGlobalObject&);
    static JSC::Structure* createStructure(JSC::VM&, JSC::JSGlobalObject&, JSC::JSValue prototype);

    DECLARE_INFO;

    // Usually defined for each specialization class.
    static JSC::JSValue prototypeForStructure(JSC::VM&, const JSDOMGlobalObject&);

    static JSC::EncodedJSValue JSC_HOST_CALL_ATTRIBUTES construct(JSC::JSGlobalObject*, JSC::CallFrame*);

private:
    JSDOMBuiltinConstructor(JSC::VM& vm, JSC::Structure* structure)
        : Base(vm, structure, construct)
    {
    }

    void finishCreation(JSC::VM&, JSDOMGlobalObject&);

    JSC::Structure* getDOMStructureForJSObject(JSC::JSGlobalObject*, JSC::JSObject* newTarget);

    // Usually defined for each specialization class.
    void initializeProperties(JSC::VM&, JSDOMGlobalObject&) { }
    // Must be defined for each specialization class.
    JSC::FunctionExecutable* initializeExecutable(JSC::VM&);
};

template<typename JSClass> inline JSDOMBuiltinConstructor<JSClass>* JSDOMBuiltinConstructor<JSClass>::create(JSC::VM& vm, JSC::Structure* structure, JSDOMGlobalObject& globalObject)
{
    JSDOMBuiltinConstructor* constructor = new (NotNull, JSC::allocateCell<JSDOMBuiltinConstructor>(vm)) JSDOMBuiltinConstructor(vm, structure);
    constructor->finishCreation(vm, globalObject);
    return constructor;
}

template<typename JSClass> inline JSC::Structure* JSDOMBuiltinConstructor<JSClass>::createStructure(JSC::VM& vm, JSC::JSGlobalObject& globalObject, JSC::JSValue prototype)
{
    auto* structure = JSC::Structure::create(vm, &globalObject, prototype, JSC::TypeInfo(JSC::InternalFunctionType, StructureFlags), info());
    structure->setMayBePrototype(true);
    return structure;
}

template<typename JSClass> inline void JSDOMBuiltinConstructor<JSClass>::finishCreation(JSC::VM& vm, JSDOMGlobalObject& globalObject)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    setInitializeFunction(vm, *JSC::JSFunction::create(vm, &globalObject, initializeExecutable(vm), &globalObject));
    initializeProperties(vm, globalObject);
}

template<typename JSClass> inline JSC::Structure* JSDOMBuiltinConstructor<JSClass>::getDOMStructureForJSObject(JSC::JSGlobalObject* lexicalGlobalObject, JSC::JSObject* newTarget)
{
    auto& vm = JSC::getVM(lexicalGlobalObject);

    if (LIKELY(newTarget == this))
        return getDOMStructure<JSClass>(vm, *globalObject());

    auto scope = DECLARE_THROW_SCOPE(vm);
    auto* newTargetGlobalObject = JSC::getFunctionRealm(lexicalGlobalObject, newTarget);
    RETURN_IF_EXCEPTION(scope, nullptr);
    auto* baseStructure = getDOMStructure<JSClass>(vm, *JSC::jsCast<JSDOMGlobalObject*>(newTargetGlobalObject));
    RELEASE_AND_RETURN(scope, JSC::InternalFunction::createSubclassStructure(lexicalGlobalObject, newTarget, baseStructure));
}

template<typename JSClass> inline JSC::EncodedJSValue JSC_HOST_CALL_ATTRIBUTES JSDOMBuiltinConstructor<JSClass>::construct(JSC::JSGlobalObject* lexicalGlobalObject, JSC::CallFrame* callFrame)
{
    ASSERT(callFrame);
    auto* castedThis = JSC::jsCast<JSDOMBuiltinConstructor*>(callFrame->jsCallee());
    auto* structure = castedThis->getDOMStructureForJSObject(lexicalGlobalObject, asObject(callFrame->newTarget()));
    if (UNLIKELY(!structure))
        return { };

    auto* jsObject = JSClass::create(structure, castedThis->globalObject());
    JSC::call(lexicalGlobalObject, castedThis->initializeFunction(), jsObject, JSC::ArgList(callFrame), "This error should never occur: initialize function is guaranteed to be callable."_s);
    return JSC::JSValue::encode(jsObject);
}

} // namespace WebCore
