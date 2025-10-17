/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 20, 2021.
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
#include "TemporalDurationConstructor.h"

#include "JSCInlines.h"
#include "TemporalDuration.h"
#include "TemporalDurationPrototype.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(TemporalDurationConstructor);

static JSC_DECLARE_HOST_FUNCTION(temporalDurationConstructorFuncFrom);
static JSC_DECLARE_HOST_FUNCTION(temporalDurationConstructorFuncCompare);

}

#include "TemporalDurationConstructor.lut.h"

namespace JSC {

const ClassInfo TemporalDurationConstructor::s_info = { "Function"_s, &Base::s_info, &temporalDurationConstructorTable, nullptr, CREATE_METHOD_TABLE(TemporalDurationConstructor) };

/* Source for TemporalDurationConstructor.lut.h
@begin temporalDurationConstructorTable
  from             temporalDurationConstructorFuncFrom             DontEnum|Function 1
  compare          temporalDurationConstructorFuncCompare          DontEnum|Function 2
@end
*/

TemporalDurationConstructor* TemporalDurationConstructor::create(VM& vm, Structure* structure, TemporalDurationPrototype* durationPrototype)
{
    auto* constructor = new (NotNull, allocateCell<TemporalDurationConstructor>(vm)) TemporalDurationConstructor(vm, structure);
    constructor->finishCreation(vm, durationPrototype);
    return constructor;
}

Structure* TemporalDurationConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

static JSC_DECLARE_HOST_FUNCTION(callTemporalDuration);
static JSC_DECLARE_HOST_FUNCTION(constructTemporalDuration);

TemporalDurationConstructor::TemporalDurationConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, callTemporalDuration, constructTemporalDuration)
{
}

void TemporalDurationConstructor::finishCreation(VM& vm, TemporalDurationPrototype* durationPrototype)
{
    Base::finishCreation(vm, 0, "Duration"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, durationPrototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
    durationPrototype->putDirectWithoutTransition(vm, vm.propertyNames->constructor, this, static_cast<unsigned>(PropertyAttribute::DontEnum));
}

JSC_DEFINE_HOST_FUNCTION(constructTemporalDuration, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* structure = JSC_GET_DERIVED_STRUCTURE(vm, durationStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    ISO8601::Duration result;
    auto count = std::min<size_t>(callFrame->argumentCount(), numberOfTemporalUnits);
    for (size_t i = 0; i < count; i++) {
        JSValue value = callFrame->uncheckedArgument(i);
        if (value.isUndefined())
            continue;

        result[i] = value.toNumber(globalObject) + 0.0;
        RETURN_IF_EXCEPTION(scope, { });

        if (!isInteger(result[i]))
            return throwVMRangeError(globalObject, scope, "Temporal.Duration properties must be integers"_s);
    }

    RELEASE_AND_RETURN(scope, JSValue::encode(TemporalDuration::tryCreateIfValid(globalObject, WTFMove(result), structure)));
}

JSC_DEFINE_HOST_FUNCTION(callTemporalDuration, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    return JSValue::encode(throwConstructorCannotBeCalledAsFunctionTypeError(globalObject, scope, "Duration"_s));
}

JSC_DEFINE_HOST_FUNCTION(temporalDurationConstructorFuncFrom, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return JSValue::encode(TemporalDuration::from(globalObject, callFrame->argument(0)));
}

JSC_DEFINE_HOST_FUNCTION(temporalDurationConstructorFuncCompare, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return JSValue::encode(TemporalDuration::compare(globalObject, callFrame->argument(0), callFrame->argument(1)));
}

} // namespace JSC
