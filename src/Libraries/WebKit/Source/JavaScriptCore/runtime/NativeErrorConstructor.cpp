/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 31, 2024.
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
#include "config.h"
#include "NativeErrorConstructor.h"

#include "ErrorInstance.h"
#include "JSCInlines.h"
#include "NativeErrorPrototype.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(NativeErrorConstructorBase);

const ClassInfo NativeErrorConstructorBase::s_info = { "Function"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(NativeErrorConstructorBase) };

static JSC_DECLARE_HOST_FUNCTION(callEvalError);
static JSC_DECLARE_HOST_FUNCTION(constructEvalError);
static JSC_DECLARE_HOST_FUNCTION(callRangeError);
static JSC_DECLARE_HOST_FUNCTION(constructRangeError);
static JSC_DECLARE_HOST_FUNCTION(callReferenceError);
static JSC_DECLARE_HOST_FUNCTION(constructReferenceError);
static JSC_DECLARE_HOST_FUNCTION(callSyntaxError);
static JSC_DECLARE_HOST_FUNCTION(constructSyntaxError);
static JSC_DECLARE_HOST_FUNCTION(callTypeError);
static JSC_DECLARE_HOST_FUNCTION(constructTypeError);
static JSC_DECLARE_HOST_FUNCTION(callURIError);
static JSC_DECLARE_HOST_FUNCTION(constructURIError);

template<ErrorType errorType>
inline EncodedJSValue NativeErrorConstructor<errorType>::constructImpl(JSGlobalObject* globalObject, CallFrame* callFrame)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    JSValue message = callFrame->argument(0);
    JSValue options = callFrame->argument(1);

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* errorStructure = JSC_GET_DERIVED_STRUCTURE(vm, errorStructureWithErrorType<errorType>, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });
    RELEASE_AND_RETURN(scope, JSValue::encode(ErrorInstance::create(globalObject, errorStructure, message, options, nullptr, TypeNothing, errorType, false)));
}

template<ErrorType errorType>
inline EncodedJSValue NativeErrorConstructor<errorType>::callImpl(JSGlobalObject* globalObject, CallFrame* callFrame)
{
    JSValue message = callFrame->argument(0);
    JSValue options = callFrame->argument(1);
    Structure* errorStructure = globalObject->errorStructure(errorType);
    return JSValue::encode(ErrorInstance::create(globalObject, errorStructure, message, options, nullptr, TypeNothing, errorType, false));
}

JSC_DEFINE_HOST_FUNCTION(callEvalError, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return NativeErrorConstructor<ErrorType::EvalError>::callImpl(globalObject, callFrame);
}
JSC_DEFINE_HOST_FUNCTION(constructEvalError, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return NativeErrorConstructor<ErrorType::EvalError>::constructImpl(globalObject, callFrame);
}

JSC_DEFINE_HOST_FUNCTION(callRangeError, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return NativeErrorConstructor<ErrorType::RangeError>::callImpl(globalObject, callFrame);
}
JSC_DEFINE_HOST_FUNCTION(constructRangeError, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return NativeErrorConstructor<ErrorType::RangeError>::constructImpl(globalObject, callFrame);
}

JSC_DEFINE_HOST_FUNCTION(callReferenceError, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return NativeErrorConstructor<ErrorType::ReferenceError>::callImpl(globalObject, callFrame);
}
JSC_DEFINE_HOST_FUNCTION(constructReferenceError, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return NativeErrorConstructor<ErrorType::ReferenceError>::constructImpl(globalObject, callFrame);
}

JSC_DEFINE_HOST_FUNCTION(callSyntaxError, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return NativeErrorConstructor<ErrorType::SyntaxError>::callImpl(globalObject, callFrame);
}
JSC_DEFINE_HOST_FUNCTION(constructSyntaxError, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return NativeErrorConstructor<ErrorType::SyntaxError>::constructImpl(globalObject, callFrame);
}

JSC_DEFINE_HOST_FUNCTION(callTypeError, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return NativeErrorConstructor<ErrorType::TypeError>::callImpl(globalObject, callFrame);
}
JSC_DEFINE_HOST_FUNCTION(constructTypeError, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return NativeErrorConstructor<ErrorType::TypeError>::constructImpl(globalObject, callFrame);
}

JSC_DEFINE_HOST_FUNCTION(callURIError, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return NativeErrorConstructor<ErrorType::URIError>::callImpl(globalObject, callFrame);
}
JSC_DEFINE_HOST_FUNCTION(constructURIError, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return NativeErrorConstructor<ErrorType::URIError>::constructImpl(globalObject, callFrame);
}

static constexpr auto callFunction(ErrorType errorType) -> decltype(&callEvalError)
{
    switch (errorType) {
    case ErrorType::EvalError: return callEvalError;
    case ErrorType::RangeError: return callRangeError;
    case ErrorType::ReferenceError: return callReferenceError;
    case ErrorType::SyntaxError: return callSyntaxError;
    case ErrorType::TypeError: return callTypeError;
    case ErrorType::URIError: return callURIError;
    default: return nullptr;
    }
}

static constexpr auto constructFunction(ErrorType errorType) -> decltype(&constructEvalError)
{
    switch (errorType) {
    case ErrorType::EvalError: return constructEvalError;
    case ErrorType::RangeError: return constructRangeError;
    case ErrorType::ReferenceError: return constructReferenceError;
    case ErrorType::SyntaxError: return constructSyntaxError;
    case ErrorType::TypeError: return constructTypeError;
    case ErrorType::URIError: return constructURIError;
    default: return nullptr;
    }
}

template<ErrorType errorType>
NativeErrorConstructor<errorType>::NativeErrorConstructor(VM& vm, Structure* structure)
    : NativeErrorConstructorBase(vm, structure, callFunction(errorType), constructFunction(errorType))
{
}

void NativeErrorConstructorBase::finishCreation(VM& vm, NativeErrorPrototype* prototype, ErrorType errorType)
{
    Base::finishCreation(vm, 1, errorTypeName(errorType), PropertyAdditionMode::WithoutStructureTransition);
    ASSERT(inherits(info()));
    
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, prototype, PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly | PropertyAttribute::DontEnum);
}

template class NativeErrorConstructor<ErrorType::EvalError>;
template class NativeErrorConstructor<ErrorType::RangeError>;
template class NativeErrorConstructor<ErrorType::ReferenceError>;
template class NativeErrorConstructor<ErrorType::SyntaxError>;
template class NativeErrorConstructor<ErrorType::TypeError>;
template class NativeErrorConstructor<ErrorType::URIError>;

} // namespace JSC
