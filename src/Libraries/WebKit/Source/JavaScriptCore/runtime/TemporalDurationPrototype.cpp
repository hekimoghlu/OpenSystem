/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 4, 2023.
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
#include "TemporalDurationPrototype.h"

#include "JSCInlines.h"
#include "TemporalDuration.h"

namespace JSC {

static JSC_DECLARE_HOST_FUNCTION(temporalDurationPrototypeFuncWith);
static JSC_DECLARE_HOST_FUNCTION(temporalDurationPrototypeFuncNegated);
static JSC_DECLARE_HOST_FUNCTION(temporalDurationPrototypeFuncAbs);
static JSC_DECLARE_HOST_FUNCTION(temporalDurationPrototypeFuncAdd);
static JSC_DECLARE_HOST_FUNCTION(temporalDurationPrototypeFuncSubtract);
static JSC_DECLARE_HOST_FUNCTION(temporalDurationPrototypeFuncRound);
static JSC_DECLARE_HOST_FUNCTION(temporalDurationPrototypeFuncTotal);
static JSC_DECLARE_HOST_FUNCTION(temporalDurationPrototypeFuncToString);
static JSC_DECLARE_HOST_FUNCTION(temporalDurationPrototypeFuncToJSON);
static JSC_DECLARE_HOST_FUNCTION(temporalDurationPrototypeFuncToLocaleString);
static JSC_DECLARE_HOST_FUNCTION(temporalDurationPrototypeFuncValueOf);
static JSC_DECLARE_CUSTOM_GETTER(temporalDurationPrototypeGetterYears);
static JSC_DECLARE_CUSTOM_GETTER(temporalDurationPrototypeGetterMonths);
static JSC_DECLARE_CUSTOM_GETTER(temporalDurationPrototypeGetterWeeks);
static JSC_DECLARE_CUSTOM_GETTER(temporalDurationPrototypeGetterDays);
static JSC_DECLARE_CUSTOM_GETTER(temporalDurationPrototypeGetterHours);
static JSC_DECLARE_CUSTOM_GETTER(temporalDurationPrototypeGetterMinutes);
static JSC_DECLARE_CUSTOM_GETTER(temporalDurationPrototypeGetterSeconds);
static JSC_DECLARE_CUSTOM_GETTER(temporalDurationPrototypeGetterMilliseconds);
static JSC_DECLARE_CUSTOM_GETTER(temporalDurationPrototypeGetterMicroseconds);
static JSC_DECLARE_CUSTOM_GETTER(temporalDurationPrototypeGetterNanoseconds);
static JSC_DECLARE_CUSTOM_GETTER(temporalDurationPrototypeGetterSign);
static JSC_DECLARE_CUSTOM_GETTER(temporalDurationPrototypeGetterBlank);

}

#include "TemporalDurationPrototype.lut.h"

namespace JSC {

const ClassInfo TemporalDurationPrototype::s_info = { "Temporal.Duration"_s, &Base::s_info, &durationPrototypeTable, nullptr, CREATE_METHOD_TABLE(TemporalDurationPrototype) };

/* Source for TemporalDurationPrototype.lut.h
@begin durationPrototypeTable
  with             temporalDurationPrototypeFuncWith               DontEnum|Function 1
  negated          temporalDurationPrototypeFuncNegated            DontEnum|Function 0
  abs              temporalDurationPrototypeFuncAbs                DontEnum|Function 0
  add              temporalDurationPrototypeFuncAdd                DontEnum|Function 1
  subtract         temporalDurationPrototypeFuncSubtract           DontEnum|Function 1
  round            temporalDurationPrototypeFuncRound              DontEnum|Function 1
  total            temporalDurationPrototypeFuncTotal              DontEnum|Function 1
  toString         temporalDurationPrototypeFuncToString           DontEnum|Function 0
  toJSON           temporalDurationPrototypeFuncToJSON             DontEnum|Function 0
  toLocaleString   temporalDurationPrototypeFuncToLocaleString     DontEnum|Function 0
  valueOf          temporalDurationPrototypeFuncValueOf            DontEnum|Function 0
  years            temporalDurationPrototypeGetterYears            DontEnum|ReadOnly|CustomAccessor
  months           temporalDurationPrototypeGetterMonths           DontEnum|ReadOnly|CustomAccessor
  weeks            temporalDurationPrototypeGetterWeeks            DontEnum|ReadOnly|CustomAccessor
  days             temporalDurationPrototypeGetterDays             DontEnum|ReadOnly|CustomAccessor
  hours            temporalDurationPrototypeGetterHours            DontEnum|ReadOnly|CustomAccessor
  minutes          temporalDurationPrototypeGetterMinutes          DontEnum|ReadOnly|CustomAccessor
  seconds          temporalDurationPrototypeGetterSeconds          DontEnum|ReadOnly|CustomAccessor
  milliseconds     temporalDurationPrototypeGetterMilliseconds     DontEnum|ReadOnly|CustomAccessor
  microseconds     temporalDurationPrototypeGetterMicroseconds     DontEnum|ReadOnly|CustomAccessor
  nanoseconds      temporalDurationPrototypeGetterNanoseconds      DontEnum|ReadOnly|CustomAccessor
  sign             temporalDurationPrototypeGetterSign             DontEnum|ReadOnly|CustomAccessor
  blank            temporalDurationPrototypeGetterBlank            DontEnum|ReadOnly|CustomAccessor
@end
*/

TemporalDurationPrototype* TemporalDurationPrototype::create(VM& vm, Structure* structure)
{
    auto* prototype = new (NotNull, allocateCell<TemporalDurationPrototype>(vm)) TemporalDurationPrototype(vm, structure);
    prototype->finishCreation(vm);
    return prototype;
}

Structure* TemporalDurationPrototype::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
}

TemporalDurationPrototype::TemporalDurationPrototype(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

void TemporalDurationPrototype::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

JSC_DEFINE_HOST_FUNCTION(temporalDurationPrototypeFuncWith, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* duration = jsDynamicCast<TemporalDuration*>(callFrame->thisValue());
    if (!duration)
        return throwVMTypeError(globalObject, scope, "Temporal.Duration.prototype.with called on value that's not a Duration"_s);
    
    JSValue durationLike = callFrame->argument(0);
    if (!durationLike.isObject())
        return throwVMTypeError(globalObject, scope, "First argument to Temporal.Duration.prototype.with must be an object"_s);

    auto result = duration->with(globalObject, asObject(durationLike));
    RETURN_IF_EXCEPTION(scope, { });

    RELEASE_AND_RETURN(scope, JSValue::encode(TemporalDuration::tryCreateIfValid(globalObject, WTFMove(result))));
}

JSC_DEFINE_HOST_FUNCTION(temporalDurationPrototypeFuncNegated, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* duration = jsDynamicCast<TemporalDuration*>(callFrame->thisValue());
    if (!duration)
        return throwVMTypeError(globalObject, scope, "Temporal.Duration.prototype.negated called on value that's not a Duration"_s);

    return JSValue::encode(TemporalDuration::create(vm, globalObject->durationStructure(), duration->negated()));
}

JSC_DEFINE_HOST_FUNCTION(temporalDurationPrototypeFuncAbs, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* duration = jsDynamicCast<TemporalDuration*>(callFrame->thisValue());
    if (!duration)
        return throwVMTypeError(globalObject, scope, "Temporal.Duration.prototype.abs called on value that's not a Duration"_s);

    return JSValue::encode(TemporalDuration::create(vm, globalObject->durationStructure(), duration->abs()));
}

JSC_DEFINE_HOST_FUNCTION(temporalDurationPrototypeFuncAdd, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* duration = jsDynamicCast<TemporalDuration*>(callFrame->thisValue());
    if (!duration)
        return throwVMTypeError(globalObject, scope, "Temporal.Duration.prototype.add called on value that's not a Duration"_s);

    auto result = duration->add(globalObject, callFrame->argument(0));
    RETURN_IF_EXCEPTION(scope, { });

    RELEASE_AND_RETURN(scope, JSValue::encode(TemporalDuration::tryCreateIfValid(globalObject, WTFMove(result))));
}

JSC_DEFINE_HOST_FUNCTION(temporalDurationPrototypeFuncSubtract, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* duration = jsDynamicCast<TemporalDuration*>(callFrame->thisValue());
    if (!duration)
        return throwVMTypeError(globalObject, scope, "Temporal.Duration.prototype.subtract called on value that's not a Duration"_s);

    auto result = duration->subtract(globalObject, callFrame->argument(0));
    RETURN_IF_EXCEPTION(scope, { });

    RELEASE_AND_RETURN(scope, JSValue::encode(TemporalDuration::tryCreateIfValid(globalObject, WTFMove(result))));
}

JSC_DEFINE_HOST_FUNCTION(temporalDurationPrototypeFuncRound, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* duration = jsDynamicCast<TemporalDuration*>(callFrame->thisValue());
    if (!duration)
        return throwVMTypeError(globalObject, scope, "Temporal.Duration.prototype.round called on value that's not a Duration"_s);

    auto options = callFrame->argument(0);
    if (options.isUndefined())
        return throwVMTypeError(globalObject, scope, "Temporal.Duration.prototype.round requires an options argument"_s);

    auto result = duration->round(globalObject, options);
    RETURN_IF_EXCEPTION(scope, { });

    RELEASE_AND_RETURN(scope, JSValue::encode(TemporalDuration::tryCreateIfValid(globalObject, WTFMove(result))));
}

JSC_DEFINE_HOST_FUNCTION(temporalDurationPrototypeFuncTotal, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* duration = jsDynamicCast<TemporalDuration*>(callFrame->thisValue());
    if (!duration)
        return throwVMTypeError(globalObject, scope, "Temporal.Duration.prototype.total called on value that's not a Duration"_s);

    auto options = callFrame->argument(0);
    if (options.isUndefined())
        return throwVMTypeError(globalObject, scope, "Temporal.Duration.prototype.total requires an options argument"_s);

    RELEASE_AND_RETURN(scope, JSValue::encode(jsNumber(duration->total(globalObject, options))));
}

JSC_DEFINE_HOST_FUNCTION(temporalDurationPrototypeFuncToString, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* duration = jsDynamicCast<TemporalDuration*>(callFrame->thisValue());
    if (!duration)
        return throwVMTypeError(globalObject, scope, "Temporal.Duration.prototype.toString called on value that's not a Duration"_s);

    RELEASE_AND_RETURN(scope, JSValue::encode(jsString(vm, duration->toString(globalObject, callFrame->argument(0)))));
}

JSC_DEFINE_HOST_FUNCTION(temporalDurationPrototypeFuncToJSON, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* duration = jsDynamicCast<TemporalDuration*>(callFrame->thisValue());
    if (!duration)
        return throwVMTypeError(globalObject, scope, "Temporal.Duration.prototype.toJSON called on value that's not a Duration"_s);

    RELEASE_AND_RETURN(scope, JSValue::encode(jsString(vm, duration->toString(globalObject))));
}

// This will be part of the ECMA-402 Intl.DurationFormat proposal; until then, we just follow ECMA-262 in mimicking toJSON.
JSC_DEFINE_HOST_FUNCTION(temporalDurationPrototypeFuncToLocaleString, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* duration = jsDynamicCast<TemporalDuration*>(callFrame->thisValue());
    if (!duration)
        return throwVMTypeError(globalObject, scope, "Temporal.Duration.prototype.toLocaleString called on value that's not a Duration"_s);

    RELEASE_AND_RETURN(scope, JSValue::encode(jsString(vm, duration->toString(globalObject))));
}

JSC_DEFINE_HOST_FUNCTION(temporalDurationPrototypeFuncValueOf, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    return throwVMTypeError(globalObject, scope, "Temporal.Duration.prototype.valueOf must not be called. To compare Duration values, use Temporal.Duration.compare"_s);
}

JSC_DEFINE_CUSTOM_GETTER(temporalDurationPrototypeGetterYears, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* duration = jsDynamicCast<TemporalDuration*>(JSValue::decode(thisValue));
    if (!duration)
        return throwVMTypeError(globalObject, scope, "Temporal.Duration.prototype.years called on value that's not a Duration"_s);

    return JSValue::encode(jsNumber(duration->years()));
}

JSC_DEFINE_CUSTOM_GETTER(temporalDurationPrototypeGetterMonths, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* duration = jsDynamicCast<TemporalDuration*>(JSValue::decode(thisValue));
    if (!duration)
        return throwVMTypeError(globalObject, scope, "Temporal.Duration.prototype.months called on value that's not a Duration"_s);

    return JSValue::encode(jsNumber(duration->months()));
}

JSC_DEFINE_CUSTOM_GETTER(temporalDurationPrototypeGetterWeeks, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* duration = jsDynamicCast<TemporalDuration*>(JSValue::decode(thisValue));
    if (!duration)
        return throwVMTypeError(globalObject, scope, "Temporal.Duration.prototype.weeks called on value that's not a Duration"_s);

    return JSValue::encode(jsNumber(duration->weeks()));
}

JSC_DEFINE_CUSTOM_GETTER(temporalDurationPrototypeGetterDays, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* duration = jsDynamicCast<TemporalDuration*>(JSValue::decode(thisValue));
    if (!duration)
        return throwVMTypeError(globalObject, scope, "Temporal.Duration.prototype.days called on value that's not a Duration"_s);

    return JSValue::encode(jsNumber(duration->days()));
}

JSC_DEFINE_CUSTOM_GETTER(temporalDurationPrototypeGetterHours, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* duration = jsDynamicCast<TemporalDuration*>(JSValue::decode(thisValue));
    if (!duration)
        return throwVMTypeError(globalObject, scope, "Temporal.Duration.prototype.hours called on value that's not a Duration"_s);

    return JSValue::encode(jsNumber(duration->hours()));
}

JSC_DEFINE_CUSTOM_GETTER(temporalDurationPrototypeGetterMinutes, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* duration = jsDynamicCast<TemporalDuration*>(JSValue::decode(thisValue));
    if (!duration)
        return throwVMTypeError(globalObject, scope, "Temporal.Duration.prototype.minutes called on value that's not a Duration"_s);

    return JSValue::encode(jsNumber(duration->minutes()));
}

JSC_DEFINE_CUSTOM_GETTER(temporalDurationPrototypeGetterSeconds, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* duration = jsDynamicCast<TemporalDuration*>(JSValue::decode(thisValue));
    if (!duration)
        return throwVMTypeError(globalObject, scope, "Temporal.Duration.prototype.seconds called on value that's not a Duration"_s);

    return JSValue::encode(jsNumber(duration->seconds()));
}

JSC_DEFINE_CUSTOM_GETTER(temporalDurationPrototypeGetterMilliseconds, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* duration = jsDynamicCast<TemporalDuration*>(JSValue::decode(thisValue));
    if (!duration)
        return throwVMTypeError(globalObject, scope, "Temporal.Duration.prototype.milliseconds called on value that's not a Duration"_s);

    return JSValue::encode(jsNumber(duration->milliseconds()));
}

JSC_DEFINE_CUSTOM_GETTER(temporalDurationPrototypeGetterMicroseconds, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* duration = jsDynamicCast<TemporalDuration*>(JSValue::decode(thisValue));
    if (!duration)
        return throwVMTypeError(globalObject, scope, "Temporal.Duration.prototype.microseconds called on value that's not a Duration"_s);

    return JSValue::encode(jsNumber(duration->microseconds()));
}

JSC_DEFINE_CUSTOM_GETTER(temporalDurationPrototypeGetterNanoseconds, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* duration = jsDynamicCast<TemporalDuration*>(JSValue::decode(thisValue));
    if (!duration)
        return throwVMTypeError(globalObject, scope, "Temporal.Duration.prototype.nanoseconds called on value that's not a Duration"_s);

    return JSValue::encode(jsNumber(duration->nanoseconds()));
}

JSC_DEFINE_CUSTOM_GETTER(temporalDurationPrototypeGetterSign, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* duration = jsDynamicCast<TemporalDuration*>(JSValue::decode(thisValue));
    if (!duration)
        return throwVMTypeError(globalObject, scope, "Temporal.Duration.prototype.sign called on value that's not a Duration"_s);

    return JSValue::encode(jsNumber(duration->sign()));
}

JSC_DEFINE_CUSTOM_GETTER(temporalDurationPrototypeGetterBlank, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* duration = jsDynamicCast<TemporalDuration*>(JSValue::decode(thisValue));
    if (!duration)
        return throwVMTypeError(globalObject, scope, "Temporal.Duration.prototype.blank called on value that's not a Duration"_s);

    return JSValue::encode(jsBoolean(!duration->sign()));
}

} // namespace JSC
