/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 17, 2023.
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
#include "TemporalPlainTimeConstructor.h"

#include "IntlObjectInlines.h"
#include "JSCInlines.h"
#include "TemporalPlainTime.h"
#include "TemporalPlainTimePrototype.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(TemporalPlainTimeConstructor);

static JSC_DECLARE_HOST_FUNCTION(temporalPlainTimeConstructorFuncFrom);
static JSC_DECLARE_HOST_FUNCTION(temporalPlainTimeConstructorFuncCompare);

}

#include "TemporalPlainTimeConstructor.lut.h"

namespace JSC {

const ClassInfo TemporalPlainTimeConstructor::s_info = { "Function"_s, &Base::s_info, &temporalPlainTimeConstructorTable, nullptr, CREATE_METHOD_TABLE(TemporalPlainTimeConstructor) };

/* Source for TemporalPlainTimeConstructor.lut.h
@begin temporalPlainTimeConstructorTable
  from             temporalPlainTimeConstructorFuncFrom             DontEnum|Function 1
  compare          temporalPlainTimeConstructorFuncCompare          DontEnum|Function 2
@end
*/

TemporalPlainTimeConstructor* TemporalPlainTimeConstructor::create(VM& vm, Structure* structure, TemporalPlainTimePrototype* plainTimePrototype)
{
    auto* constructor = new (NotNull, allocateCell<TemporalPlainTimeConstructor>(vm)) TemporalPlainTimeConstructor(vm, structure);
    constructor->finishCreation(vm, plainTimePrototype);
    return constructor;
}

Structure* TemporalPlainTimeConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

static JSC_DECLARE_HOST_FUNCTION(callTemporalPlainTime);
static JSC_DECLARE_HOST_FUNCTION(constructTemporalPlainTime);

TemporalPlainTimeConstructor::TemporalPlainTimeConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, callTemporalPlainTime, constructTemporalPlainTime)
{
}

void TemporalPlainTimeConstructor::finishCreation(VM& vm, TemporalPlainTimePrototype* plainTimePrototype)
{
    Base::finishCreation(vm, 0, "PlainTime"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, plainTimePrototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
    plainTimePrototype->putDirectWithoutTransition(vm, vm.propertyNames->constructor, this, static_cast<unsigned>(PropertyAttribute::DontEnum));
}

JSC_DEFINE_HOST_FUNCTION(constructTemporalPlainTime, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* structure = JSC_GET_DERIVED_STRUCTURE(vm, plainTimeStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    ISO8601::Duration duration { };
    auto count = std::min<size_t>(callFrame->argumentCount(), numberOfTemporalPlainTimeUnits);
    for (unsigned i = 0; i < count; i++) {
        unsigned durationIndex = i + static_cast<unsigned>(TemporalUnit::Hour);
        duration[durationIndex] = callFrame->uncheckedArgument(i).toIntegerOrInfinity(globalObject);
        RETURN_IF_EXCEPTION(scope, { });
        if (!std::isfinite(duration[durationIndex]))
            return throwVMRangeError(globalObject, scope, "Temporal.PlainTime properties must be finite"_s);
    }
    RELEASE_AND_RETURN(scope, JSValue::encode(TemporalPlainTime::tryCreateIfValid(globalObject, structure, WTFMove(duration))));
}

JSC_DEFINE_HOST_FUNCTION(callTemporalPlainTime, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    return JSValue::encode(throwConstructorCannotBeCalledAsFunctionTypeError(globalObject, scope, "PlainTime"_s));
}

// https://tc39.es/proposal-temporal/#sec-temporal.plaintime.from
JSC_DEFINE_HOST_FUNCTION(temporalPlainTimeConstructorFuncFrom, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSObject* options = intlGetOptionsObject(globalObject, callFrame->argument(1));
    RETURN_IF_EXCEPTION(scope, { });

    TemporalOverflow overflow = toTemporalOverflow(globalObject, options);
    RETURN_IF_EXCEPTION(scope, { });

    JSValue itemValue = callFrame->argument(0);

    if (itemValue.inherits<TemporalPlainTime>())
        RELEASE_AND_RETURN(scope, JSValue::encode(TemporalPlainTime::create(vm, globalObject->plainTimeStructure(), jsCast<TemporalPlainTime*>(itemValue)->plainTime())));

    RELEASE_AND_RETURN(scope, JSValue::encode(TemporalPlainTime::from(globalObject, itemValue, overflow)));
}

// https://tc39.es/proposal-temporal/#sec-temporal.plaintime.compare
JSC_DEFINE_HOST_FUNCTION(temporalPlainTimeConstructorFuncCompare, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* one = TemporalPlainTime::from(globalObject, callFrame->argument(0), std::nullopt);
    RETURN_IF_EXCEPTION(scope, { });

    auto* two = TemporalPlainTime::from(globalObject, callFrame->argument(1), std::nullopt);
    RETURN_IF_EXCEPTION(scope, { });

    return JSValue::encode(jsNumber(TemporalPlainTime::compare(one->plainTime(), two->plainTime())));
}

} // namespace JSC
