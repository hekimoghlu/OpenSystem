/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 22, 2024.
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
#include "IntlDateTimeFormatConstructor.h"

#include "IntlDateTimeFormat.h"
#include "IntlDateTimeFormatPrototype.h"
#include "IntlObjectInlines.h"
#include "JSCInlines.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(IntlDateTimeFormatConstructor);

static JSC_DECLARE_HOST_FUNCTION(intlDateTimeFormatConstructorFuncSupportedLocalesOf);

}

#include "IntlDateTimeFormatConstructor.lut.h"

namespace JSC {

const ClassInfo IntlDateTimeFormatConstructor::s_info = { "Function"_s, &InternalFunction::s_info, &dateTimeFormatConstructorTable, nullptr, CREATE_METHOD_TABLE(IntlDateTimeFormatConstructor) };

/* Source for IntlDateTimeFormatConstructor.lut.h
@begin dateTimeFormatConstructorTable
  supportedLocalesOf             intlDateTimeFormatConstructorFuncSupportedLocalesOf             DontEnum|Function 1
@end
*/

IntlDateTimeFormatConstructor* IntlDateTimeFormatConstructor::create(VM& vm, Structure* structure, IntlDateTimeFormatPrototype* dateTimeFormatPrototype)
{
    IntlDateTimeFormatConstructor* constructor = new (NotNull, allocateCell<IntlDateTimeFormatConstructor>(vm)) IntlDateTimeFormatConstructor(vm, structure);
    constructor->finishCreation(vm, dateTimeFormatPrototype);
    return constructor;
}

Structure* IntlDateTimeFormatConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

static JSC_DECLARE_HOST_FUNCTION(callIntlDateTimeFormat);
static JSC_DECLARE_HOST_FUNCTION(constructIntlDateTimeFormat);

IntlDateTimeFormatConstructor::IntlDateTimeFormatConstructor(VM& vm, Structure* structure)
    : InternalFunction(vm, structure, callIntlDateTimeFormat, constructIntlDateTimeFormat)
{
}

void IntlDateTimeFormatConstructor::finishCreation(VM& vm, IntlDateTimeFormatPrototype* dateTimeFormatPrototype)
{
    Base::finishCreation(vm, 0, "DateTimeFormat"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, dateTimeFormatPrototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
}

JSC_DEFINE_HOST_FUNCTION(constructIntlDateTimeFormat, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    // 12.1.2 Intl.DateTimeFormat ([locales [, options]]) (ECMA-402 2.0)
    // 1. If NewTarget is undefined, let newTarget be the active function object, else let newTarget be NewTarget.
    // 2. Let dateTimeFormat be OrdinaryCreateFromConstructor(newTarget, %DateTimeFormatPrototype%).
    // 3. ReturnIfAbrupt(dateTimeFormat).
    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* structure = JSC_GET_DERIVED_STRUCTURE(vm, dateTimeFormatStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    IntlDateTimeFormat* dateTimeFormat = IntlDateTimeFormat::create(vm, structure);
    ASSERT(dateTimeFormat);

    // 4. Return InitializeDateTimeFormat(dateTimeFormat, locales, options).
    scope.release();
    dateTimeFormat->initializeDateTimeFormat(globalObject, callFrame->argument(0), callFrame->argument(1), IntlDateTimeFormat::RequiredComponent::Any, IntlDateTimeFormat::Defaults::Date);
    return JSValue::encode(dateTimeFormat);
}

JSC_DEFINE_HOST_FUNCTION(callIntlDateTimeFormat, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    // 12.1.2 Intl.DateTimeFormat ([locales [, options]]) (ECMA-402 2.0)
    // 1. If NewTarget is undefined, let newTarget be the active function object, else let newTarget be NewTarget.
    // NewTarget is always undefined when called as a function.

    // FIXME: Workaround to provide compatibility with ECMA-402 1.0 call/apply patterns.
    // https://bugs.webkit.org/show_bug.cgi?id=153679
    return JSValue::encode(constructIntlInstanceWithWorkaroundForLegacyIntlConstructor(globalObject, callFrame->thisValue(), callFrame->jsCallee(), [&] (VM& vm) {
        // 2. Let dateTimeFormat be OrdinaryCreateFromConstructor(newTarget, %DateTimeFormatPrototype%).
        // 3. ReturnIfAbrupt(dateTimeFormat).
        IntlDateTimeFormat* dateTimeFormat = IntlDateTimeFormat::create(vm, globalObject->dateTimeFormatStructure());
        ASSERT(dateTimeFormat);

        // 4. Return InitializeDateTimeFormat(dateTimeFormat, locales, options).
        dateTimeFormat->initializeDateTimeFormat(globalObject, callFrame->argument(0), callFrame->argument(1), IntlDateTimeFormat::RequiredComponent::Any, IntlDateTimeFormat::Defaults::Date);
        return dateTimeFormat;
    }));
}

JSC_DEFINE_HOST_FUNCTION(intlDateTimeFormatConstructorFuncSupportedLocalesOf, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    // 12.2.2 Intl.DateTimeFormat.supportedLocalesOf(locales [, options]) (ECMA-402 2.0)

    // 1. Let availableLocales be %DateTimeFormat%.[[availableLocales]].
    const auto& availableLocales = intlDateTimeFormatAvailableLocales();

    // 2. Let requestedLocales be CanonicalizeLocaleList(locales).
    Vector<String> requestedLocales = canonicalizeLocaleList(globalObject, callFrame->argument(0));
    RETURN_IF_EXCEPTION(scope, encodedJSValue());

    // 3. Return SupportedLocales(availableLocales, requestedLocales, options).
    RELEASE_AND_RETURN(scope, JSValue::encode(supportedLocales(globalObject, availableLocales, requestedLocales, callFrame->argument(1))));
}

} // namespace JSC
