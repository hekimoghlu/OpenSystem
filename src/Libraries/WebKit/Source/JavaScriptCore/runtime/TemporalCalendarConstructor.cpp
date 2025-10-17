/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 7, 2025.
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
#include "TemporalCalendarConstructor.h"

#include "JSCInlines.h"
#include "TemporalCalendar.h"
#include "TemporalCalendarPrototype.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(TemporalCalendarConstructor);
static JSC_DECLARE_HOST_FUNCTION(temporalCalendarConstructorFuncFrom);

}

#include "TemporalCalendarConstructor.lut.h"

namespace JSC {

const ClassInfo TemporalCalendarConstructor::s_info = { "Function"_s, &InternalFunction::s_info, &temporalCalendarConstructorTable, nullptr, CREATE_METHOD_TABLE(TemporalCalendarConstructor) };

/* Source for TemporalCalendarConstructor.lut.h
@begin temporalCalendarConstructorTable
    from             temporalCalendarConstructorFuncFrom             DontEnum|Function 1
@end
*/

TemporalCalendarConstructor* TemporalCalendarConstructor::create(VM& vm, Structure* structure, TemporalCalendarPrototype* temporalCalendarPrototype)
{
    TemporalCalendarConstructor* constructor = new (NotNull, allocateCell<TemporalCalendarConstructor>(vm)) TemporalCalendarConstructor(vm, structure);
    constructor->finishCreation(vm, temporalCalendarPrototype);
    return constructor;
}

Structure* TemporalCalendarConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

static JSC_DECLARE_HOST_FUNCTION(callTemporalCalendar);
static JSC_DECLARE_HOST_FUNCTION(constructTemporalCalendar);

TemporalCalendarConstructor::TemporalCalendarConstructor(VM& vm, Structure* structure)
    : InternalFunction(vm, structure, callTemporalCalendar, constructTemporalCalendar)
{
}

void TemporalCalendarConstructor::finishCreation(VM& vm, TemporalCalendarPrototype* temporalCalendarPrototype)
{
    Base::finishCreation(vm, 0, "Calendar"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, temporalCalendarPrototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
    temporalCalendarPrototype->putDirectWithoutTransition(vm, vm.propertyNames->constructor, this, static_cast<unsigned>(PropertyAttribute::DontEnum));
}

JSC_DEFINE_HOST_FUNCTION(constructTemporalCalendar, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* structure = JSC_GET_DERIVED_STRUCTURE(vm, calendarStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    auto calendarString = callFrame->argument(0).toWTFString(globalObject);
    RETURN_IF_EXCEPTION(scope, { });

    std::optional<CalendarID> identifier = TemporalCalendar::isBuiltinCalendar(calendarString);
    if (!identifier) {
        throwRangeError(globalObject, scope, "invalid calendar ID"_s);
        return { };
    }

    return JSValue::encode(TemporalCalendar::create(vm, structure, identifier.value()));
}

JSC_DEFINE_HOST_FUNCTION(callTemporalCalendar, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    return JSValue::encode(throwConstructorCannotBeCalledAsFunctionTypeError(globalObject, scope, "Calendar"_s));
}

JSC_DEFINE_HOST_FUNCTION(temporalCalendarConstructorFuncFrom, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return JSValue::encode(TemporalCalendar::from(globalObject, callFrame->argument(0)));
}

} // namespace JSC
