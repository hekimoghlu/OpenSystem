/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 17, 2023.
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
#include "IntlDurationFormatPrototype.h"

#include "IntlDurationFormat.h"
#include "JSCInlines.h"
#include "TemporalDuration.h"

namespace JSC {

static JSC_DECLARE_HOST_FUNCTION(intlDurationFormatPrototypeFuncFormat);
static JSC_DECLARE_HOST_FUNCTION(intlDurationFormatPrototypeFuncFormatToParts);
static JSC_DECLARE_HOST_FUNCTION(intlDurationFormatPrototypeFuncResolvedOptions);

}

#include "IntlDurationFormatPrototype.lut.h"

namespace JSC {

const ClassInfo IntlDurationFormatPrototype::s_info = { "Intl.DurationFormat"_s, &Base::s_info, &durationFormatPrototypeTable, nullptr, CREATE_METHOD_TABLE(IntlDurationFormatPrototype) };

/* Source for IntlDurationFormatPrototype.lut.h
@begin durationFormatPrototypeTable
  format           intlDurationFormatPrototypeFuncFormat             DontEnum|Function 1
  formatToParts    intlDurationFormatPrototypeFuncFormatToParts      DontEnum|Function 1
  resolvedOptions  intlDurationFormatPrototypeFuncResolvedOptions    DontEnum|Function 0
@end
*/

IntlDurationFormatPrototype* IntlDurationFormatPrototype::create(VM& vm, Structure* structure)
{
    auto* object = new (NotNull, allocateCell<IntlDurationFormatPrototype>(vm)) IntlDurationFormatPrototype(vm, structure);
    object->finishCreation(vm);
    return object;
}

Structure* IntlDurationFormatPrototype::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
}

IntlDurationFormatPrototype::IntlDurationFormatPrototype(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

void IntlDurationFormatPrototype::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

// https://tc39.es/proposal-intl-duration-format/#sec-Intl.DurationFormat.prototype.format
JSC_DEFINE_HOST_FUNCTION(intlDurationFormatPrototypeFuncFormat, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* durationFormat = jsDynamicCast<IntlDurationFormat*>(callFrame->thisValue());
    if (UNLIKELY(!durationFormat))
        return throwVMTypeError(globalObject, scope, "Intl.DurationFormat.prototype.format called on value that's not a DurationFormat"_s);

    JSValue argument = callFrame->argument(0);
    if (UNLIKELY(!argument.isObject() && !argument.isString()))
        return throwVMTypeError(globalObject, scope, "Intl.DurationFormat.prototype.format argument needs to be an object or a string"_s);

    auto duration = TemporalDuration::toISO8601Duration(globalObject, argument);
    RETURN_IF_EXCEPTION(scope, { });

    RELEASE_AND_RETURN(scope, JSValue::encode(durationFormat->format(globalObject, WTFMove(duration))));
}

// https://tc39.es/proposal-intl-duration-format/#sec-Intl.DurationFormat.prototype.formatToParts
JSC_DEFINE_HOST_FUNCTION(intlDurationFormatPrototypeFuncFormatToParts, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* durationFormat = jsDynamicCast<IntlDurationFormat*>(callFrame->thisValue());
    if (UNLIKELY(!durationFormat))
        return throwVMTypeError(globalObject, scope, "Intl.DurationFormat.prototype.formatToParts called on value that's not a DurationFormat"_s);

    JSValue argument = callFrame->argument(0);
    if (UNLIKELY(!argument.isObject() && !argument.isString()))
        return throwVMTypeError(globalObject, scope, "Intl.DurationFormat.prototype.formatToParts argument needs to be an object or a string"_s);

    auto duration = TemporalDuration::toISO8601Duration(globalObject, argument);
    RETURN_IF_EXCEPTION(scope, { });

    RELEASE_AND_RETURN(scope, JSValue::encode(durationFormat->formatToParts(globalObject, WTFMove(duration))));
}

// https://tc39.es/proposal-intl-duration-format/#sec-Intl.DurationFormat.prototype.resolvedOptions
JSC_DEFINE_HOST_FUNCTION(intlDurationFormatPrototypeFuncResolvedOptions, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* durationFormat = jsDynamicCast<IntlDurationFormat*>(callFrame->thisValue());
    if (UNLIKELY(!durationFormat))
        return throwVMTypeError(globalObject, scope, "Intl.DurationFormat.prototype.resolvedOptions called on value that's not a DurationFormat"_s);

    RELEASE_AND_RETURN(scope, JSValue::encode(durationFormat->resolvedOptions(globalObject)));
}

} // namespace JSC
