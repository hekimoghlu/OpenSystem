/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 8, 2023.
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
#include "TemporalInstantConstructor.h"

#include "JSCInlines.h"
#include "TemporalInstant.h"
#include "TemporalInstantPrototype.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(TemporalInstantConstructor);

static JSC_DECLARE_HOST_FUNCTION(temporalInstantConstructorFuncFrom);
static JSC_DECLARE_HOST_FUNCTION(temporalInstantConstructorFuncFromEpochMilliseconds);
static JSC_DECLARE_HOST_FUNCTION(temporalInstantConstructorFuncFromEpochNanoseconds);
static JSC_DECLARE_HOST_FUNCTION(temporalInstantConstructorFuncCompare);

}

#include "TemporalInstantConstructor.lut.h"

namespace JSC {

const ClassInfo TemporalInstantConstructor::s_info = { "Function"_s, &Base::s_info, &temporalInstantConstructorTable, nullptr, CREATE_METHOD_TABLE(TemporalInstantConstructor) };

/* Source for TemporalInstantConstructor.lut.h
@begin temporalInstantConstructorTable
  from                   temporalInstantConstructorFuncFrom                   DontEnum|Function 1
  fromEpochMilliseconds  temporalInstantConstructorFuncFromEpochMilliseconds  DontEnum|Function 1
  fromEpochNanoseconds   temporalInstantConstructorFuncFromEpochNanoseconds   DontEnum|Function 1
  compare                temporalInstantConstructorFuncCompare                DontEnum|Function 2
@end
*/

TemporalInstantConstructor* TemporalInstantConstructor::create(VM& vm, Structure* structure, TemporalInstantPrototype* instantPrototype)
{
    auto* constructor = new (NotNull, allocateCell<TemporalInstantConstructor>(vm)) TemporalInstantConstructor(vm, structure);
    constructor->finishCreation(vm, instantPrototype);
    return constructor;
}

Structure* TemporalInstantConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

static JSC_DECLARE_HOST_FUNCTION(callTemporalInstant);
static JSC_DECLARE_HOST_FUNCTION(constructTemporalInstant);

TemporalInstantConstructor::TemporalInstantConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, callTemporalInstant, constructTemporalInstant)
{
}

void TemporalInstantConstructor::finishCreation(VM& vm, TemporalInstantPrototype* instantPrototype)
{
    Base::finishCreation(vm, 1, "Instant"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, instantPrototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
    instantPrototype->putDirectWithoutTransition(vm, vm.propertyNames->constructor, this, static_cast<unsigned>(PropertyAttribute::DontEnum));
}

JSC_DEFINE_HOST_FUNCTION(constructTemporalInstant, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* structure = JSC_GET_DERIVED_STRUCTURE(vm, instantStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    if (callFrame->argumentCount() < 1)
        return throwVMTypeError(globalObject, scope, "Missing required epochNanoseconds argument to Temporal.Instant"_s);

    RELEASE_AND_RETURN(scope, JSValue::encode(TemporalInstant::tryCreateIfValid(globalObject, callFrame->uncheckedArgument(0), structure)));
}

JSC_DEFINE_HOST_FUNCTION(callTemporalInstant, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    return JSValue::encode(throwConstructorCannotBeCalledAsFunctionTypeError(globalObject, scope, "Instant"_s));
}

JSC_DEFINE_HOST_FUNCTION(temporalInstantConstructorFuncFrom, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return JSValue::encode(TemporalInstant::from(globalObject, callFrame->argument(0)));
}

JSC_DEFINE_HOST_FUNCTION(temporalInstantConstructorFuncFromEpochMilliseconds, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return JSValue::encode(TemporalInstant::fromEpochMilliseconds(globalObject, callFrame->argument(0)));
}

JSC_DEFINE_HOST_FUNCTION(temporalInstantConstructorFuncFromEpochNanoseconds, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return JSValue::encode(TemporalInstant::fromEpochNanoseconds(globalObject, callFrame->argument(0)));
}

JSC_DEFINE_HOST_FUNCTION(temporalInstantConstructorFuncCompare, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return JSValue::encode(TemporalInstant::compare(globalObject, callFrame->argument(0), callFrame->argument(1)));
}

} // namespace JSC
