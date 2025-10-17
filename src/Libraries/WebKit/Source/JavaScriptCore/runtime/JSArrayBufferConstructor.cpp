/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 27, 2023.
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
#include "JSArrayBufferConstructor.h"

#include "BuiltinNames.h"
#include "JSArrayBuffer.h"
#include "JSArrayBufferPrototype.h"
#include "JSArrayBufferView.h"
#include "JSCInlines.h"

namespace JSC {

static JSC_DECLARE_HOST_FUNCTION(arrayBufferFuncIsView);
static JSC_DECLARE_HOST_FUNCTION(callArrayBuffer);
static JSC_DECLARE_HOST_FUNCTION(constructArrayBuffer);
static JSC_DECLARE_HOST_FUNCTION(constructSharedArrayBuffer);

template<>
const ClassInfo JSArrayBufferConstructor::s_info = {
    "Function"_s, &Base::s_info, nullptr, nullptr,
    CREATE_METHOD_TABLE(JSArrayBufferConstructor)
};

template<>
const ClassInfo JSSharedArrayBufferConstructor::s_info = {
    "Function"_s, &Base::s_info, nullptr, nullptr,
    CREATE_METHOD_TABLE(JSSharedArrayBufferConstructor)
};

template<ArrayBufferSharingMode sharingMode>
JSGenericArrayBufferConstructor<sharingMode>::JSGenericArrayBufferConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, callArrayBuffer, sharingMode == ArrayBufferSharingMode::Default ? constructArrayBuffer : constructSharedArrayBuffer)
{
}

template<ArrayBufferSharingMode sharingMode>
void JSGenericArrayBufferConstructor<sharingMode>::finishCreation(VM& vm, JSArrayBufferPrototype* prototype)
{
    Base::finishCreation(vm, 1, arrayBufferSharingModeName(sharingMode), PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, prototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);

    JSGlobalObject* globalObject = this->globalObject();
    putDirectNonIndexAccessorWithoutTransition(vm, vm.propertyNames->speciesSymbol, globalObject->arrayBufferSpeciesGetterSetter(sharingMode), PropertyAttribute::Accessor | PropertyAttribute::ReadOnly | PropertyAttribute::DontEnum);

    if (sharingMode == ArrayBufferSharingMode::Default) {
        JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->isView, arrayBufferFuncIsView, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Public);
        JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->builtinNames().isViewPrivateName(), arrayBufferFuncIsView, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Public);
    }
}

template<ArrayBufferSharingMode sharingMode>
EncodedJSValue JSGenericArrayBufferConstructor<sharingMode>::constructImpl(JSGlobalObject* globalObject, CallFrame* callFrame)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    double lengthDouble = 0;
    std::optional<size_t> maxByteLength;

    bool hasArguments = callFrame->argumentCount();
    if (hasArguments) {
        lengthDouble = callFrame->uncheckedArgument(0).toNumber(globalObject);
        RETURN_IF_EXCEPTION(scope, { });
        JSValue options = callFrame->argument(1);
        if (options.isObject()) {
            JSValue maxByteLengthValue = asObject(options)->get(globalObject, vm.propertyNames->maxByteLength);
            RETURN_IF_EXCEPTION(scope, { });
            if (!maxByteLengthValue.isUndefined()) {
                maxByteLength = maxByteLengthValue.toTypedArrayIndex(globalObject, "maxByteLength"_s);
                RETURN_IF_EXCEPTION(scope, { });
            }
        }
    }

    // https://tc39.es/proposal-resizablearraybuffer/#sec-allocatesharedarraybuffer
    RefPtr<ArrayBuffer> buffer;
    if (maxByteLength) {
        if (maxByteLength.value() < lengthDouble)
            return throwVMRangeError(globalObject, scope, "ArrayBuffer length exceeds maxByteLength option"_s);
    }

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* structure = JSC_GET_DERIVED_STRUCTURE(vm, arrayBufferStructureWithSharingMode<sharingMode>, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    size_t length = 0;
    if (hasArguments) {
        JSValue lengthDoubleValue = JSValue(JSValue::EncodeAsDouble, lengthDouble);
        length = lengthDoubleValue.toTypedArrayIndex(globalObject, "length"_s);
        RETURN_IF_EXCEPTION(scope, { });
    }

    if (maxByteLength) {
        if constexpr (sharingMode == ArrayBufferSharingMode::Shared) {
            buffer = ArrayBuffer::tryCreateShared(vm, length, 1, maxByteLength.value());
            if (!buffer)
                return JSValue::encode(throwOutOfMemoryError(globalObject, scope));
        }
    }

    if (!buffer) {
        buffer = ArrayBuffer::tryCreate(length, 1, maxByteLength);
        if (!buffer)
            return JSValue::encode(throwOutOfMemoryError(globalObject, scope));
        if constexpr (sharingMode == ArrayBufferSharingMode::Shared)
            buffer->makeShared();
    }

    ASSERT(sharingMode == buffer->sharingMode());

    return JSValue::encode(JSArrayBuffer::create(vm, structure, WTFMove(buffer)));
}

template<ArrayBufferSharingMode sharingMode>
Structure* JSGenericArrayBufferConstructor<sharingMode>::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

template<ArrayBufferSharingMode sharingMode>
const ClassInfo* JSGenericArrayBufferConstructor<sharingMode>::info()
{
    return &JSGenericArrayBufferConstructor<sharingMode>::s_info;
}

JSC_DEFINE_HOST_FUNCTION(callArrayBuffer, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    return JSValue::encode(throwConstructorCannotBeCalledAsFunctionTypeError(globalObject, scope, "ArrayBuffer"_s));
}

JSC_DEFINE_HOST_FUNCTION(constructArrayBuffer, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return JSGenericArrayBufferConstructor<ArrayBufferSharingMode::Default>::constructImpl(globalObject, callFrame);
}

JSC_DEFINE_HOST_FUNCTION(constructSharedArrayBuffer, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return JSGenericArrayBufferConstructor<ArrayBufferSharingMode::Shared>::constructImpl(globalObject, callFrame);
}

// ------------------------------ Functions --------------------------------

// ECMA 24.1.3.1
JSC_DEFINE_HOST_FUNCTION(arrayBufferFuncIsView, (JSGlobalObject*, CallFrame* callFrame))
{
    return JSValue::encode(jsBoolean(jsDynamicCast<JSArrayBufferView*>(callFrame->argument(0))));
}

// Instantiate JSGenericArrayBufferConstructors.
template class JSGenericArrayBufferConstructor<ArrayBufferSharingMode::Shared>;
template class JSGenericArrayBufferConstructor<ArrayBufferSharingMode::Default>;

} // namespace JSC

