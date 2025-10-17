/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 14, 2023.
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
#include "TemporalTimeZoneConstructor.h"

#include "ISO8601.h"
#include "JSCInlines.h"
#include "TemporalTimeZone.h"
#include "TemporalTimeZonePrototype.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(TemporalTimeZoneConstructor);
static JSC_DECLARE_HOST_FUNCTION(temporalTimeZoneConstructorFuncFrom);

}

#include "TemporalTimeZoneConstructor.lut.h"

namespace JSC {

const ClassInfo TemporalTimeZoneConstructor::s_info = { "Function"_s, &InternalFunction::s_info, &temporalTimeZoneConstructorTable, nullptr, CREATE_METHOD_TABLE(TemporalTimeZoneConstructor) };

/* Source for TemporalTimeZoneConstructor.lut.h
@begin temporalTimeZoneConstructorTable
    from             temporalTimeZoneConstructorFuncFrom             DontEnum|Function 1
@end
*/

TemporalTimeZoneConstructor* TemporalTimeZoneConstructor::create(VM& vm, Structure* structure, TemporalTimeZonePrototype* temporalTimeZonePrototype)
{
    TemporalTimeZoneConstructor* constructor = new (NotNull, allocateCell<TemporalTimeZoneConstructor>(vm)) TemporalTimeZoneConstructor(vm, structure);
    constructor->finishCreation(vm, temporalTimeZonePrototype);
    return constructor;
}

Structure* TemporalTimeZoneConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

static JSC_DECLARE_HOST_FUNCTION(callTemporalTimeZone);
static JSC_DECLARE_HOST_FUNCTION(constructTemporalTimeZone);

TemporalTimeZoneConstructor::TemporalTimeZoneConstructor(VM& vm, Structure* structure)
    : InternalFunction(vm, structure, callTemporalTimeZone, constructTemporalTimeZone)
{
}

void TemporalTimeZoneConstructor::finishCreation(VM& vm, TemporalTimeZonePrototype* temporalTimeZonePrototype)
{
    Base::finishCreation(vm, 0, "TimeZone"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, temporalTimeZonePrototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
    temporalTimeZonePrototype->putDirectWithoutTransition(vm, vm.propertyNames->constructor, this, static_cast<unsigned>(PropertyAttribute::DontEnum));
}

JSC_DEFINE_HOST_FUNCTION(constructTemporalTimeZone, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* structure = JSC_GET_DERIVED_STRUCTURE(vm, timeZoneStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    auto timeZoneString = callFrame->argument(0).toWTFString(globalObject);
    RETURN_IF_EXCEPTION(scope, { });

    std::optional<int64_t> utcOffset = ISO8601::parseUTCOffset(timeZoneString);
    if (utcOffset)
        return JSValue::encode(TemporalTimeZone::createFromUTCOffset(vm, structure, utcOffset.value()));

    std::optional<TimeZoneID> identifier = ISO8601::parseTimeZoneName(timeZoneString);
    if (!identifier) {
        throwRangeError(globalObject, scope, "argument needs to be UTC offset string or TimeZone identifier"_s);
        return { };
    }
    return JSValue::encode(TemporalTimeZone::createFromID(vm, structure, identifier.value()));
}

JSC_DEFINE_HOST_FUNCTION(callTemporalTimeZone, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    return JSValue::encode(throwConstructorCannotBeCalledAsFunctionTypeError(globalObject, scope, "TimeZone"_s));
}

JSC_DEFINE_HOST_FUNCTION(temporalTimeZoneConstructorFuncFrom, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return JSValue::encode(TemporalTimeZone::from(globalObject, callFrame->argument(0)));
}

} // namespace JSC
