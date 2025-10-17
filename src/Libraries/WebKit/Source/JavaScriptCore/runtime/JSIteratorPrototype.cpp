/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 16, 2025.
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
#include "JSIteratorPrototype.h"

#include "BuiltinNames.h"
#include "CachedCall.h"
#include "InterpreterInlines.h"
#include "IteratorOperations.h"
#include "JSCBuiltins.h"
#include "JSCInlines.h"
#include "JSIteratorConstructor.h"
#include "VMEntryScopeInlines.h"

namespace JSC {

const ClassInfo JSIteratorPrototype::s_info = { "Iterator"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSIteratorPrototype) };

static JSC_DECLARE_HOST_FUNCTION(iteratorProtoFuncIterator);
static JSC_DECLARE_CUSTOM_GETTER(iteratorProtoConstructorGetter);
static JSC_DECLARE_CUSTOM_SETTER(iteratorProtoConstructorSetter);
static JSC_DECLARE_CUSTOM_GETTER(iteratorProtoToStringTagGetter);
static JSC_DECLARE_CUSTOM_SETTER(iteratorProtoToStringTagSetter);
static JSC_DECLARE_HOST_FUNCTION(iteratorProtoFuncToArray);
static JSC_DECLARE_HOST_FUNCTION(iteratorProtoFuncForEach);

void JSIteratorPrototype::finishCreation(VM& vm, JSGlobalObject* globalObject)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    JSFunction* iteratorFunction = JSFunction::create(vm, globalObject, 0, "[Symbol.iterator]"_s, iteratorProtoFuncIterator, ImplementationVisibility::Public, IteratorIntrinsic);
    putDirectWithoutTransition(vm, vm.propertyNames->iteratorSymbol, iteratorFunction, static_cast<unsigned>(PropertyAttribute::DontEnum));

    if (Options::useIteratorHelpers()) {
        // https://tc39.es/proposal-iterator-helpers/#sec-iteratorprototype.constructor
        putDirectCustomGetterSetterWithoutTransition(vm, vm.propertyNames->constructor, CustomGetterSetter::create(vm, iteratorProtoConstructorGetter, iteratorProtoConstructorSetter), static_cast<unsigned>(PropertyAttribute::DontEnum | PropertyAttribute::CustomAccessor));
        // https://tc39.es/proposal-iterator-helpers/#sec-iteratorprototype-@@tostringtag
        putDirectCustomGetterSetterWithoutTransition(vm, vm.propertyNames->toStringTagSymbol, CustomGetterSetter::create(vm, iteratorProtoToStringTagGetter, iteratorProtoToStringTagSetter), static_cast<unsigned>(PropertyAttribute::DontEnum | PropertyAttribute::CustomAccessor));
        // https://tc39.es/proposal-iterator-helpers/#sec-iteratorprototype.toarray
        JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->toArray, iteratorProtoFuncToArray, static_cast<unsigned>(PropertyAttribute::DontEnum), 0, ImplementationVisibility::Private);
        // https://tc39.es/proposal-iterator-helpers/#sec-iteratorprototype.foreach
        JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->forEach, iteratorProtoFuncForEach, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Private);
        // https://tc39.es/proposal-iterator-helpers/#sec-iteratorprototype.some
        JSC_BUILTIN_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->builtinNames().somePublicName(), jsIteratorPrototypeSomeCodeGenerator, static_cast<unsigned>(PropertyAttribute::DontEnum));
        // https://tc39.es/proposal-iterator-helpers/#sec-iteratorprototype.every
        JSC_BUILTIN_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->builtinNames().everyPublicName(), jsIteratorPrototypeEveryCodeGenerator, static_cast<unsigned>(PropertyAttribute::DontEnum));
        // https://tc39.es/proposal-iterator-helpers/#sec-iteratorprototype.find
        JSC_BUILTIN_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->builtinNames().findPublicName(), jsIteratorPrototypeFindCodeGenerator, static_cast<unsigned>(PropertyAttribute::DontEnum));
        // https://tc39.es/proposal-iterator-helpers/#sec-iteratorprototype.reduce
        JSC_BUILTIN_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->builtinNames().reducePublicName(), jsIteratorPrototypeReduceCodeGenerator, static_cast<unsigned>(PropertyAttribute::DontEnum));
        // https://tc39.es/proposal-iterator-helpers/#sec-iteratorprototype.map
        JSC_BUILTIN_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->builtinNames().mapPublicName(), jsIteratorPrototypeMapCodeGenerator, static_cast<unsigned>(PropertyAttribute::DontEnum));
        // https://tc39.es/proposal-iterator-helpers/#sec-iteratorprototype.filter
        JSC_BUILTIN_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->builtinNames().filterPublicName(), jsIteratorPrototypeFilterCodeGenerator, static_cast<unsigned>(PropertyAttribute::DontEnum));
        // https://tc39.es/proposal-iterator-helpers/#sec-iteratorprototype.take
        JSC_BUILTIN_FUNCTION_WITHOUT_TRANSITION("take"_s, jsIteratorPrototypeTakeCodeGenerator, static_cast<unsigned>(PropertyAttribute::DontEnum));
        // https://tc39.es/proposal-iterator-helpers/#sec-iteratorprototype.drop
        JSC_BUILTIN_FUNCTION_WITHOUT_TRANSITION("drop"_s, jsIteratorPrototypeDropCodeGenerator, static_cast<unsigned>(PropertyAttribute::DontEnum));
        // https://tc39.es/proposal-iterator-helpers/#sec-iteratorprototype.flatmap
        JSC_BUILTIN_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->builtinNames().flatMapPublicName(), jsIteratorPrototypeFlatMapCodeGenerator, static_cast<unsigned>(PropertyAttribute::DontEnum));
    }

    if (Options::useIteratorChunking()) {
        // https://tc39.es/proposal-iterator-chunking/#sec-iterator.prototype.chunks
        JSC_BUILTIN_FUNCTION_WITHOUT_TRANSITION("chunks"_s, jsIteratorPrototypeChunksCodeGenerator, static_cast<unsigned>(PropertyAttribute::DontEnum));
        // https://tc39.es/proposal-iterator-chunking/#sec-iterator.prototype.windows
        JSC_BUILTIN_FUNCTION_WITHOUT_TRANSITION("windows"_s, jsIteratorPrototypeWindowsCodeGenerator, static_cast<unsigned>(PropertyAttribute::DontEnum));
    }
}

JSC_DEFINE_HOST_FUNCTION(iteratorProtoFuncIterator, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return JSValue::encode(callFrame->thisValue().toThis(globalObject, ECMAMode::strict()));
}

// https://tc39.es/proposal-iterator-helpers/#sec-get-iteratorprototype-constructor
JSC_DEFINE_CUSTOM_GETTER(iteratorProtoConstructorGetter, (JSGlobalObject* globalObject, EncodedJSValue, PropertyName))
{
    return JSValue::encode(globalObject->iteratorConstructor());
}

// https://tc39.es/proposal-iterator-helpers/#sec-set-iteratorprototype-constructor
JSC_DEFINE_CUSTOM_SETTER(iteratorProtoConstructorSetter, (JSGlobalObject* globalObject, EncodedJSValue encodedThisValue, EncodedJSValue value, PropertyName propertyName))
{
    bool shouldThrow = true;
    setterThatIgnoresPrototypeProperties(globalObject, JSValue::decode(encodedThisValue), globalObject->iteratorPrototype(), propertyName, JSValue::decode(value), shouldThrow);
    return JSValue::encode(jsUndefined());
}

// https://tc39.es/proposal-iterator-helpers/#sec-get-iteratorprototype-@@tostringtag
JSC_DEFINE_CUSTOM_GETTER(iteratorProtoToStringTagGetter, (JSGlobalObject* globalObject, EncodedJSValue, PropertyName))
{
    VM& vm = globalObject->vm();
    return JSValue::encode(jsNontrivialString(vm, "Iterator"_s));
}

// https://tc39.es/proposal-iterator-helpers/#sec-set-iteratorprototype-@@tostringtag
JSC_DEFINE_CUSTOM_SETTER(iteratorProtoToStringTagSetter, (JSGlobalObject* globalObject, EncodedJSValue encodedThisValue, EncodedJSValue value, PropertyName propertyName))
{
    bool shouldThrow = true;
    setterThatIgnoresPrototypeProperties(globalObject, JSValue::decode(encodedThisValue), globalObject->iteratorPrototype(), propertyName, JSValue::decode(value), shouldThrow);
    return JSValue::encode(jsUndefined());
}

// https://tc39.es/proposal-iterator-helpers/#sec-iteratorprototype.toarray
JSC_DEFINE_HOST_FUNCTION(iteratorProtoFuncToArray, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    if (!thisValue.isObject())
        return throwVMTypeError(globalObject, scope, "Iterator.prototype.toArray requires that |this| be an Object."_s);

    MarkedArgumentBuffer value;
    forEachInIteratorProtocol(globalObject, thisValue, [&value, &scope](VM&, JSGlobalObject* globalObject, JSValue nextItem) {
        value.append(nextItem);
        if (UNLIKELY(value.hasOverflowed()))
            throwOutOfMemoryError(globalObject, scope);
    });
    RETURN_IF_EXCEPTION(scope, { });

    auto* array = constructArray(globalObject, static_cast<ArrayAllocationProfile*>(nullptr), value);
    RETURN_IF_EXCEPTION(scope, { });

    return JSValue::encode(array);
}

// https://tc39.es/proposal-iterator-helpers/#sec-iteratorprototype.foreach
JSC_DEFINE_HOST_FUNCTION(iteratorProtoFuncForEach, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    if (!thisValue.isObject())
        return throwVMTypeError(globalObject, scope, "Iterator.prototype.forEach requires that |this| be an Object."_s);

    JSValue callbackArg = callFrame->argument(0);
    if (!callbackArg.isCallable())
        return throwVMTypeError(globalObject, scope, "Iterator.prototype.forEach requires the callback argument to be callable."_s);

    auto callData = JSC::getCallData(callbackArg);
    ASSERT(callData.type != CallData::Type::None);

    uint64_t counter = 0;

    if (LIKELY(callData.type == CallData::Type::JS)) {
        CachedCall cachedCall(globalObject, jsCast<JSFunction*>(callbackArg), 2);
        RETURN_IF_EXCEPTION(scope, { });

        forEachInIteratorProtocol(globalObject, thisValue, [&](VM&, JSGlobalObject*, JSValue nextItem) ALWAYS_INLINE_LAMBDA {
            cachedCall.callWithArguments(globalObject, jsUndefined(), nextItem, jsNumber(counter++));
        });
    } else {
        forEachInIteratorProtocol(globalObject, thisValue, [&](VM&, JSGlobalObject*, JSValue nextItem) ALWAYS_INLINE_LAMBDA {
            MarkedArgumentBuffer args;
            args.append(nextItem);
            args.append(jsNumber(counter++));
            ASSERT(!args.hasOverflowed());

            call(globalObject, callbackArg, callData, jsUndefined(), args);
        });
    }

    RETURN_IF_EXCEPTION(scope, { });
    return JSValue::encode(jsUndefined());
}

} // namespace JSC
