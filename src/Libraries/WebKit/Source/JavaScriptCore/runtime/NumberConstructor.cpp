/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 24, 2024.
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
#include "NumberConstructor.h"

#include "JSCInlines.h"
#include "NumberObject.h"
#include "NumberPrototype.h"

namespace JSC {

static JSC_DECLARE_HOST_FUNCTION(numberConstructorFuncIsNaN);
static JSC_DECLARE_HOST_FUNCTION(numberConstructorFuncIsInteger);
static JSC_DECLARE_HOST_FUNCTION(numberConstructorFuncIsSafeInteger);

} // namespace JSC

#include "NumberConstructor.lut.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(NumberConstructor);

const ClassInfo NumberConstructor::s_info = { "Function"_s, &Base::s_info, &numberConstructorTable, nullptr, CREATE_METHOD_TABLE(NumberConstructor) };

/* Source for NumberConstructor.lut.h
@begin numberConstructorTable
  isFinite       JSBuiltin                           DontEnum|Function 1
  isNaN          numberConstructorFuncIsNaN          DontEnum|Function 1 NumberIsNaNIntrinsic
  isSafeInteger  numberConstructorFuncIsSafeInteger  DontEnum|Function 1
@end
*/

static JSC_DECLARE_HOST_FUNCTION(callNumberConstructor);
static JSC_DECLARE_HOST_FUNCTION(constructNumberConstructor);

NumberConstructor::NumberConstructor(VM& vm, NativeExecutable* executable, JSGlobalObject* globalObject, Structure* structure)
    : Base(vm, executable, globalObject, structure)
{
}

NumberConstructor* NumberConstructor::create(VM& vm, Structure* structure, NumberPrototype* numberPrototype)
{
    JSGlobalObject* globalObject = structure->globalObject();
    NativeExecutable* executable = vm.getHostFunction(callNumberConstructor, ImplementationVisibility::Public, NumberConstructorIntrinsic, constructNumberConstructor, nullptr, vm.propertyNames->Number.string());
    NumberConstructor* constructor = new (NotNull, allocateCell<NumberConstructor>(vm)) NumberConstructor(vm, executable, globalObject, structure);
    constructor->finishCreation(vm, numberPrototype);
    return constructor;
}

void NumberConstructor::finishCreation(VM& vm, NumberPrototype* numberPrototype)
{
    ASSERT(inherits(info()));
    Base::finishCreation(vm);

    JSGlobalObject* globalObject = numberPrototype->globalObject();

    putDirectWithoutTransition(vm, vm.propertyNames->prototype, numberPrototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
    putDirectWithoutTransition(vm, vm.propertyNames->length, jsNumber(1), PropertyAttribute::DontEnum | PropertyAttribute::ReadOnly);
    putDirectWithoutTransition(vm, vm.propertyNames->name, jsString(vm, vm.propertyNames->Number.string()), PropertyAttribute::ReadOnly | PropertyAttribute::DontEnum);

    putDirectWithoutTransition(vm, Identifier::fromString(vm, "EPSILON"_s), jsDoubleNumber(std::numeric_limits<double>::epsilon()), PropertyAttribute::DontDelete | PropertyAttribute::DontEnum | PropertyAttribute::ReadOnly);
    putDirectWithoutTransition(vm, Identifier::fromString(vm, "MAX_VALUE"_s), jsDoubleNumber(1.7976931348623157E+308), PropertyAttribute::DontDelete | PropertyAttribute::DontEnum | PropertyAttribute::ReadOnly);
    putDirectWithoutTransition(vm, Identifier::fromString(vm, "MIN_VALUE"_s), jsDoubleNumber(5E-324), PropertyAttribute::DontDelete | PropertyAttribute::DontEnum | PropertyAttribute::ReadOnly);
    putDirectWithoutTransition(vm, Identifier::fromString(vm, "MAX_SAFE_INTEGER"_s), jsDoubleNumber(maxSafeInteger()), PropertyAttribute::DontDelete | PropertyAttribute::DontEnum | PropertyAttribute::ReadOnly);
    putDirectWithoutTransition(vm, Identifier::fromString(vm, "MIN_SAFE_INTEGER"_s), jsDoubleNumber(minSafeInteger()), PropertyAttribute::DontDelete | PropertyAttribute::DontEnum | PropertyAttribute::ReadOnly);
    putDirectWithoutTransition(vm, Identifier::fromString(vm, "NEGATIVE_INFINITY"_s), jsDoubleNumber(-std::numeric_limits<double>::infinity()), PropertyAttribute::DontDelete | PropertyAttribute::DontEnum | PropertyAttribute::ReadOnly);
    putDirectWithoutTransition(vm, Identifier::fromString(vm, "POSITIVE_INFINITY"_s), jsDoubleNumber(std::numeric_limits<double>::infinity()), PropertyAttribute::DontDelete | PropertyAttribute::DontEnum | PropertyAttribute::ReadOnly);
    putDirectWithoutTransition(vm, vm.propertyNames->NaN, jsNaN(), PropertyAttribute::DontDelete | PropertyAttribute::DontEnum | PropertyAttribute::ReadOnly);

    putDirectWithoutTransition(vm, vm.propertyNames->parseInt, numberPrototype->globalObject()->parseIntFunction(), static_cast<unsigned>(PropertyAttribute::DontEnum));
    putDirectWithoutTransition(vm, vm.propertyNames->parseFloat, numberPrototype->globalObject()->parseFloatFunction(), static_cast<unsigned>(PropertyAttribute::DontEnum));

    JSC_NATIVE_INTRINSIC_FUNCTION_WITHOUT_TRANSITION(Identifier::fromString(vm, "isInteger"_s), numberConstructorFuncIsInteger, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Public, NumberIsIntegerIntrinsic);
}

// ECMA 15.7.1
JSC_DEFINE_HOST_FUNCTION(constructNumberConstructor, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    double n = 0;
    if (callFrame->argumentCount()) {
        JSValue numeric = callFrame->uncheckedArgument(0).toNumeric(globalObject);
        RETURN_IF_EXCEPTION(scope, { });
        if (numeric.isNumber())
            n = numeric.asNumber();
        else {
            ASSERT(numeric.isBigInt());
            numeric = JSBigInt::toNumber(numeric);
            ASSERT(numeric.isNumber());
            n = numeric.asNumber();
        }
    }

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* structure = JSC_GET_DERIVED_STRUCTURE(vm, numberObjectStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    NumberObject* object = NumberObject::create(vm, structure);
    object->setInternalValue(vm, jsNumber(n));
    return JSValue::encode(object);
}

// ECMA 15.7.2
JSC_DEFINE_HOST_FUNCTION(callNumberConstructor, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    if (!callFrame->argumentCount())
        return JSValue::encode(jsNumber(0));
    JSValue numeric = callFrame->uncheckedArgument(0).toNumeric(globalObject);
    RETURN_IF_EXCEPTION(scope, { });
    if (numeric.isNumber())
        return JSValue::encode(numeric);
    ASSERT(numeric.isBigInt());
    return JSValue::encode(JSBigInt::toNumber(numeric));
}

// ECMA-262 20.1.2.3
JSC_DEFINE_HOST_FUNCTION(numberConstructorFuncIsInteger, (JSGlobalObject*, CallFrame* callFrame))
{
    return JSValue::encode(jsBoolean(NumberConstructor::isIntegerImpl(callFrame->argument(0))));
}

// ECMA-262 20.1.2.5
JSC_DEFINE_HOST_FUNCTION(numberConstructorFuncIsSafeInteger, (JSGlobalObject*, CallFrame* callFrame))
{
    JSValue argument = callFrame->argument(0);
    if (argument.isInt32())
        return JSValue::encode(jsBoolean(true));
    if (!argument.isDouble())
        return JSValue::encode(jsBoolean(false));
    return JSValue::encode(jsBoolean(isSafeInteger(argument.asDouble())));
}

JSC_DEFINE_HOST_FUNCTION(numberConstructorFuncIsNaN, (JSGlobalObject*, CallFrame* callFrame))
{
    JSValue argument = callFrame->argument(0);
    if (!argument.isNumber())
        return JSValue::encode(jsBoolean(false));
    return JSValue::encode(jsBoolean(std::isnan(argument.asNumber())));
}

} // namespace JSC
