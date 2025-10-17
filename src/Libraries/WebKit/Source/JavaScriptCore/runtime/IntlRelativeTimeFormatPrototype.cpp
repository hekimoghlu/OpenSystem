/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 11, 2022.
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
#include "IntlRelativeTimeFormatPrototype.h"

#include "IntlRelativeTimeFormat.h"
#include "JSCInlines.h"

namespace JSC {

static JSC_DECLARE_HOST_FUNCTION(intlRelativeTimeFormatPrototypeFuncFormat);
static JSC_DECLARE_HOST_FUNCTION(intlRelativeTimeFormatPrototypeFuncFormatToParts);
static JSC_DECLARE_HOST_FUNCTION(intlRelativeTimeFormatPrototypeFuncResolvedOptions);

}

#include "IntlRelativeTimeFormatPrototype.lut.h"

namespace JSC {

const ClassInfo IntlRelativeTimeFormatPrototype::s_info = { "Intl.RelativeTimeFormat"_s, &Base::s_info, &relativeTimeFormatPrototypeTable, nullptr, CREATE_METHOD_TABLE(IntlRelativeTimeFormatPrototype) };

/* Source for IntlRelativeTimeFormatPrototype.lut.h
@begin relativeTimeFormatPrototypeTable
  format           intlRelativeTimeFormatPrototypeFuncFormat           DontEnum|Function 2
  formatToParts    intlRelativeTimeFormatPrototypeFuncFormatToParts    DontEnum|Function 2
  resolvedOptions  intlRelativeTimeFormatPrototypeFuncResolvedOptions  DontEnum|Function 0
@end
*/

IntlRelativeTimeFormatPrototype* IntlRelativeTimeFormatPrototype::create(VM& vm, Structure* structure)
{
    auto* object = new (NotNull, allocateCell<IntlRelativeTimeFormatPrototype>(vm)) IntlRelativeTimeFormatPrototype(vm, structure);
    object->finishCreation(vm);
    return object;
}

Structure* IntlRelativeTimeFormatPrototype::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
}

IntlRelativeTimeFormatPrototype::IntlRelativeTimeFormatPrototype(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

void IntlRelativeTimeFormatPrototype::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

// https://tc39.es/ecma402/#sec-Intl.RelativeTimeFormat.prototype.format
JSC_DEFINE_HOST_FUNCTION(intlRelativeTimeFormatPrototypeFuncFormat, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* relativeTimeFormat = jsDynamicCast<IntlRelativeTimeFormat*>(callFrame->thisValue());
    if (UNLIKELY(!relativeTimeFormat))
        return JSValue::encode(throwTypeError(globalObject, scope, "Intl.RelativeTimeFormat.prototype.format called on value that's not a RelativeTimeFormat"_s));

    double value = callFrame->argument(0).toNumber(globalObject);
    RETURN_IF_EXCEPTION(scope, encodedJSValue());

    String unit = callFrame->argument(1).toWTFString(globalObject);
    RETURN_IF_EXCEPTION(scope, encodedJSValue());

    RELEASE_AND_RETURN(scope, JSValue::encode(relativeTimeFormat->format(globalObject, value, unit)));
}

// https://tc39.es/ecma402/#sec-Intl.RelativeTimeFormat.prototype.formatToParts
JSC_DEFINE_HOST_FUNCTION(intlRelativeTimeFormatPrototypeFuncFormatToParts, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* relativeTimeFormat = jsDynamicCast<IntlRelativeTimeFormat*>(callFrame->thisValue());
    if (UNLIKELY(!relativeTimeFormat))
        return JSValue::encode(throwTypeError(globalObject, scope, "Intl.RelativeTimeFormat.prototype.formatToParts called on value that's not a RelativeTimeFormat"_s));

    double value = callFrame->argument(0).toNumber(globalObject);
    RETURN_IF_EXCEPTION(scope, encodedJSValue());

    String unit = callFrame->argument(1).toWTFString(globalObject);
    RETURN_IF_EXCEPTION(scope, encodedJSValue());

    RELEASE_AND_RETURN(scope, JSValue::encode(relativeTimeFormat->formatToParts(globalObject, value, unit)));
}

// https://tc39.es/ecma402/#sec-intl.relativetimeformat.prototype.resolvedoptions
JSC_DEFINE_HOST_FUNCTION(intlRelativeTimeFormatPrototypeFuncResolvedOptions, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* relativeTimeFormat = jsDynamicCast<IntlRelativeTimeFormat*>(callFrame->thisValue());
    if (UNLIKELY(!relativeTimeFormat))
        return JSValue::encode(throwTypeError(globalObject, scope, "Intl.RelativeTimeFormat.prototype.resolvedOptions called on value that's not a RelativeTimeFormat"_s));

    RELEASE_AND_RETURN(scope, JSValue::encode(relativeTimeFormat->resolvedOptions(globalObject)));
}

} // namespace JSC
