/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 19, 2025.
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
#include "TemporalPlainDateTimeConstructor.h"

#include "IntlObjectInlines.h"
#include "JSCInlines.h"
#include "TemporalPlainDateTime.h"
#include "TemporalPlainDateTimePrototype.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(TemporalPlainDateTimeConstructor);

static JSC_DECLARE_HOST_FUNCTION(temporalPlainDateTimeConstructorFuncFrom);
static JSC_DECLARE_HOST_FUNCTION(temporalPlainDateTimeConstructorFuncCompare);

}

#include "TemporalPlainDateTimeConstructor.lut.h"

namespace JSC {

const ClassInfo TemporalPlainDateTimeConstructor::s_info = { "Function"_s, &Base::s_info, &temporalPlainDateTimeConstructorTable, nullptr, CREATE_METHOD_TABLE(TemporalPlainDateTimeConstructor) };

/* Source for TemporalPlainDateTimeConstructor.lut.h
@begin temporalPlainDateTimeConstructorTable
  from             temporalPlainDateTimeConstructorFuncFrom             DontEnum|Function 1
  compare          temporalPlainDateTimeConstructorFuncCompare          DontEnum|Function 2
@end
*/

TemporalPlainDateTimeConstructor* TemporalPlainDateTimeConstructor::create(VM& vm, Structure* structure, TemporalPlainDateTimePrototype* plainDateTimePrototype)
{
    auto* constructor = new (NotNull, allocateCell<TemporalPlainDateTimeConstructor>(vm)) TemporalPlainDateTimeConstructor(vm, structure);
    constructor->finishCreation(vm, plainDateTimePrototype);
    return constructor;
}

Structure* TemporalPlainDateTimeConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

static JSC_DECLARE_HOST_FUNCTION(callTemporalPlainDateTime);
static JSC_DECLARE_HOST_FUNCTION(constructTemporalPlainDateTime);

TemporalPlainDateTimeConstructor::TemporalPlainDateTimeConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, callTemporalPlainDateTime, constructTemporalPlainDateTime)
{
}

void TemporalPlainDateTimeConstructor::finishCreation(VM& vm, TemporalPlainDateTimePrototype* plainDateTimePrototype)
{
    Base::finishCreation(vm, 3, "PlainDateTime"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, plainDateTimePrototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
    plainDateTimePrototype->putDirectWithoutTransition(vm, vm.propertyNames->constructor, this, static_cast<unsigned>(PropertyAttribute::DontEnum));
}

JSC_DEFINE_HOST_FUNCTION(constructTemporalPlainDateTime, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* structure = JSC_GET_DERIVED_STRUCTURE(vm, plainDateTimeStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    ISO8601::Duration duration { };
    auto count = std::min<size_t>(callFrame->argumentCount(), numberOfTemporalPlainDateUnits + numberOfTemporalPlainTimeUnits);
    for (unsigned i = 0; i < count; i++) {
        unsigned durationIndex = i >= static_cast<unsigned>(TemporalUnit::Week) ? i + 1 : i;
        duration[durationIndex] = callFrame->uncheckedArgument(i).toIntegerOrInfinity(globalObject);
        RETURN_IF_EXCEPTION(scope, { });
        if (!std::isfinite(duration[durationIndex]))
            return throwVMRangeError(globalObject, scope, "Temporal.PlainDateTime properties must be finite"_s);
    }

    RELEASE_AND_RETURN(scope, JSValue::encode(TemporalPlainDateTime::tryCreateIfValid(globalObject, structure, WTFMove(duration))));
}

JSC_DEFINE_HOST_FUNCTION(callTemporalPlainDateTime, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    return JSValue::encode(throwConstructorCannotBeCalledAsFunctionTypeError(globalObject, scope, "PlainDateTime"_s));
}

// https://tc39.es/proposal-temporal/#sec-temporal.plaindatetime.from
JSC_DEFINE_HOST_FUNCTION(temporalPlainDateTimeConstructorFuncFrom, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSObject* options = intlGetOptionsObject(globalObject, callFrame->argument(1));
    RETURN_IF_EXCEPTION(scope, { });

    TemporalOverflow overflow = toTemporalOverflow(globalObject, options);
    RETURN_IF_EXCEPTION(scope, { });

    JSValue itemValue = callFrame->argument(0);

    if (itemValue.inherits<TemporalPlainDateTime>())
        RELEASE_AND_RETURN(scope, JSValue::encode(TemporalPlainDateTime::create(vm, globalObject->plainDateTimeStructure(), jsCast<TemporalPlainDateTime*>(itemValue)->plainDate(), jsCast<TemporalPlainDateTime*>(itemValue)->plainTime())));

    RELEASE_AND_RETURN(scope, JSValue::encode(TemporalPlainDateTime::from(globalObject, itemValue, overflow)));
}

// https://tc39.es/proposal-temporal/#sec-temporal.plaindatetime.compare
JSC_DEFINE_HOST_FUNCTION(temporalPlainDateTimeConstructorFuncCompare, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* one = TemporalPlainDateTime::from(globalObject, callFrame->argument(0), std::nullopt);
    RETURN_IF_EXCEPTION(scope, { });

    auto* two = TemporalPlainDateTime::from(globalObject, callFrame->argument(1), std::nullopt);
    RETURN_IF_EXCEPTION(scope, { });

    return JSValue::encode(jsNumber(TemporalPlainDateTime::compare(one, two)));
}

} // namespace JSC
