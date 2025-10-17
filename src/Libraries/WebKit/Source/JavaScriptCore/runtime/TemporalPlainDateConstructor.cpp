/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 5, 2023.
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
#include "TemporalPlainDateConstructor.h"

#include "IntlObjectInlines.h"
#include "JSCInlines.h"
#include "TemporalPlainDate.h"
#include "TemporalPlainDatePrototype.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(TemporalPlainDateConstructor);

static JSC_DECLARE_HOST_FUNCTION(temporalPlainDateConstructorFuncFrom);
static JSC_DECLARE_HOST_FUNCTION(temporalPlainDateConstructorFuncCompare);

}

#include "TemporalPlainDateConstructor.lut.h"

namespace JSC {

const ClassInfo TemporalPlainDateConstructor::s_info = { "Function"_s, &Base::s_info, &temporalPlainDateConstructorTable, nullptr, CREATE_METHOD_TABLE(TemporalPlainDateConstructor) };

/* Source for TemporalPlainDateConstructor.lut.h
@begin temporalPlainDateConstructorTable
  from             temporalPlainDateConstructorFuncFrom             DontEnum|Function 1
  compare          temporalPlainDateConstructorFuncCompare          DontEnum|Function 2
@end
*/

TemporalPlainDateConstructor* TemporalPlainDateConstructor::create(VM& vm, Structure* structure, TemporalPlainDatePrototype* plainDatePrototype)
{
    auto* constructor = new (NotNull, allocateCell<TemporalPlainDateConstructor>(vm)) TemporalPlainDateConstructor(vm, structure);
    constructor->finishCreation(vm, plainDatePrototype);
    return constructor;
}

Structure* TemporalPlainDateConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

static JSC_DECLARE_HOST_FUNCTION(callTemporalPlainDate);
static JSC_DECLARE_HOST_FUNCTION(constructTemporalPlainDate);

TemporalPlainDateConstructor::TemporalPlainDateConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, callTemporalPlainDate, constructTemporalPlainDate)
{
}

void TemporalPlainDateConstructor::finishCreation(VM& vm, TemporalPlainDatePrototype* plainDatePrototype)
{
    Base::finishCreation(vm, 3, "PlainDate"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, plainDatePrototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
    plainDatePrototype->putDirectWithoutTransition(vm, vm.propertyNames->constructor, this, static_cast<unsigned>(PropertyAttribute::DontEnum));
}

JSC_DEFINE_HOST_FUNCTION(constructTemporalPlainDate, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* structure = JSC_GET_DERIVED_STRUCTURE(vm, plainDateStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    ISO8601::Duration duration { };
    auto argumentCount = callFrame->argumentCount();

    if (argumentCount > 0) {
        auto value = callFrame->uncheckedArgument(0).toIntegerWithTruncation(globalObject);
        if (!std::isfinite(value))
            return throwVMRangeError(globalObject, scope, "Temporal.PlainDate year property must be finite"_s);
        duration.setYears(value);
        RETURN_IF_EXCEPTION(scope, { });
    }

    if (argumentCount > 1) {
        auto value = callFrame->uncheckedArgument(1).toIntegerWithTruncation(globalObject);
        if (!std::isfinite(value))
            return throwVMRangeError(globalObject, scope, "Temporal.PlainDate month property must be finite"_s);
        duration.setMonths(value);
        RETURN_IF_EXCEPTION(scope, { });
    }

    if (argumentCount > 2) {
        auto value = callFrame->uncheckedArgument(2).toIntegerWithTruncation(globalObject);
        if (!std::isfinite(value))
            return throwVMRangeError(globalObject, scope, "Temporal.PlainDate day property must be finite"_s);
        duration.setDays(value);
        RETURN_IF_EXCEPTION(scope, { });
    }

    RELEASE_AND_RETURN(scope, JSValue::encode(TemporalPlainDate::tryCreateIfValid(globalObject, structure, WTFMove(duration))));
}

JSC_DEFINE_HOST_FUNCTION(callTemporalPlainDate, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    return JSValue::encode(throwConstructorCannotBeCalledAsFunctionTypeError(globalObject, scope, "PlainDate"_s));
}

// https://tc39.es/proposal-temporal/#sec-temporal.plaindate.from
JSC_DEFINE_HOST_FUNCTION(temporalPlainDateConstructorFuncFrom, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSObject* options = intlGetOptionsObject(globalObject, callFrame->argument(1));
    RETURN_IF_EXCEPTION(scope, { });

    TemporalOverflow overflow = toTemporalOverflow(globalObject, options);
    RETURN_IF_EXCEPTION(scope, { });

    JSValue itemValue = callFrame->argument(0);

    if (itemValue.inherits<TemporalPlainDate>())
        RELEASE_AND_RETURN(scope, JSValue::encode(TemporalPlainDate::create(vm, globalObject->plainDateStructure(), jsCast<TemporalPlainDate*>(itemValue)->plainDate())));

    RELEASE_AND_RETURN(scope, JSValue::encode(TemporalPlainDate::from(globalObject, itemValue, overflow)));
}

// https://tc39.es/proposal-temporal/#sec-temporal.plaindate.compare
JSC_DEFINE_HOST_FUNCTION(temporalPlainDateConstructorFuncCompare, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* one = TemporalPlainDate::from(globalObject, callFrame->argument(0), std::nullopt);
    RETURN_IF_EXCEPTION(scope, { });

    auto* two = TemporalPlainDate::from(globalObject, callFrame->argument(1), std::nullopt);
    RETURN_IF_EXCEPTION(scope, { });

    return JSValue::encode(jsNumber(TemporalCalendar::isoDateCompare(one->plainDate(), two->plainDate())));
}

} // namespace JSC
