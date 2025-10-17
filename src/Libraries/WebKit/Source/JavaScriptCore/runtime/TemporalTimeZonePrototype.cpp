/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 30, 2022.
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
#include "TemporalTimeZonePrototype.h"

#include "BuiltinNames.h"
#include "ISO8601.h"
#include "JSCInlines.h"
#include "TemporalTimeZone.h"

namespace JSC {

static JSC_DECLARE_HOST_FUNCTION(temporalTimeZonePrototypeFuncToString);
static JSC_DECLARE_HOST_FUNCTION(temporalTimeZonePrototypeFuncToJSON);
static JSC_DECLARE_CUSTOM_GETTER(temporalTimeZonePrototypeGetterId);

}

#include "TemporalTimeZonePrototype.lut.h"

namespace JSC {

const ClassInfo TemporalTimeZonePrototype::s_info = { "Temporal.TimeZone"_s, &Base::s_info, &temporalTimeZonePrototypeTable, nullptr, CREATE_METHOD_TABLE(TemporalTimeZonePrototype) };

/* Source for TemporalTimeZonePrototype.lut.h
@begin temporalTimeZonePrototypeTable
    toString        temporalTimeZonePrototypeFuncToString     DontEnum|Function 0
    toJSON          temporalTimeZonePrototypeFuncToJSON       DontEnum|Function 0
    id              temporalTimeZonePrototypeGetterId         ReadOnly|DontEnum|CustomAccessor
@end
*/

TemporalTimeZonePrototype* TemporalTimeZonePrototype::create(VM& vm, JSGlobalObject* globalObject, Structure* structure)
{
    TemporalTimeZonePrototype* object = new (NotNull, allocateCell<TemporalTimeZonePrototype>(vm)) TemporalTimeZonePrototype(vm, structure);
    object->finishCreation(vm, globalObject);
    return object;
}

Structure* TemporalTimeZonePrototype::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
}

TemporalTimeZonePrototype::TemporalTimeZonePrototype(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

void TemporalTimeZonePrototype::finishCreation(VM& vm, JSGlobalObject*)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

// https://tc39.es/proposal-temporal/#sec-get-temporal.timezone.prototype.id
JSC_DEFINE_CUSTOM_GETTER(temporalTimeZonePrototypeGetterId, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    return JSValue::encode(JSValue::decode(thisValue).toString(globalObject));
}

// https://tc39.es/proposal-temporal/#sec-temporal.timezone.prototype.tostring
JSC_DEFINE_HOST_FUNCTION(temporalTimeZonePrototypeFuncToString, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* timeZone = jsDynamicCast<TemporalTimeZone*>(callFrame->thisValue());
    if (!timeZone)
        return throwVMTypeError(globalObject, scope, "Temporal.TimeZone.prototype.toString called on value that's not a TimeZone"_s);

    auto variant = timeZone->timeZone();
    auto string = WTF::switchOn(variant,
        [](TimeZoneID identifier) -> String {
            return intlAvailableTimeZones()[identifier];
        },
        [](int64_t offset) -> String {
            return ISO8601::formatTimeZoneOffsetString(offset);
        });
    return JSValue::encode(jsString(vm, WTFMove(string)));
}

// https://tc39.es/proposal-temporal/#sec-temporal.timezone.prototype.tojson
JSC_DEFINE_HOST_FUNCTION(temporalTimeZonePrototypeFuncToJSON, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return JSValue::encode(callFrame->thisValue().toString(globalObject));
}

} // namespace JSC
