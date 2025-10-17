/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 24, 2025.
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
#include "IntlNumberFormatConstructor.h"

#include "IntlNumberFormat.h"
#include "IntlNumberFormatPrototype.h"
#include "IntlObjectInlines.h"
#include "JSCInlines.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(IntlNumberFormatConstructor);

static JSC_DECLARE_HOST_FUNCTION(intlNumberFormatConstructorFuncSupportedLocalesOf);

}

#include "IntlNumberFormatConstructor.lut.h"

namespace JSC {

const ClassInfo IntlNumberFormatConstructor::s_info = { "Function"_s, &Base::s_info, &numberFormatConstructorTable, nullptr, CREATE_METHOD_TABLE(IntlNumberFormatConstructor) };

/* Source for IntlNumberFormatConstructor.lut.h
@begin numberFormatConstructorTable
  supportedLocalesOf             intlNumberFormatConstructorFuncSupportedLocalesOf             DontEnum|Function 1
@end
*/

IntlNumberFormatConstructor* IntlNumberFormatConstructor::create(VM& vm, Structure* structure, IntlNumberFormatPrototype* numberFormatPrototype)
{
    IntlNumberFormatConstructor* constructor = new (NotNull, allocateCell<IntlNumberFormatConstructor>(vm)) IntlNumberFormatConstructor(vm, structure);
    constructor->finishCreation(vm, numberFormatPrototype);
    return constructor;
}

Structure* IntlNumberFormatConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

static JSC_DECLARE_HOST_FUNCTION(callIntlNumberFormat);
static JSC_DECLARE_HOST_FUNCTION(constructIntlNumberFormat);

IntlNumberFormatConstructor::IntlNumberFormatConstructor(VM& vm, Structure* structure)
    : InternalFunction(vm, structure, callIntlNumberFormat, constructIntlNumberFormat)
{
}

void IntlNumberFormatConstructor::finishCreation(VM& vm, IntlNumberFormatPrototype* numberFormatPrototype)
{
    Base::finishCreation(vm, 0, "NumberFormat"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, numberFormatPrototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
}

JSC_DEFINE_HOST_FUNCTION(constructIntlNumberFormat, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    // 11.1.2 Intl.NumberFormat ([locales [, options]]) (ECMA-402 2.0)
    // 1. If NewTarget is undefined, let newTarget be the active function object, else let newTarget be NewTarget.
    // 2. Let numberFormat be OrdinaryCreateFromConstructor(newTarget, %NumberFormatPrototype%).
    // 3. ReturnIfAbrupt(numberFormat).
    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* structure = JSC_GET_DERIVED_STRUCTURE(vm, numberFormatStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    IntlNumberFormat* numberFormat = IntlNumberFormat::create(vm, structure);
    ASSERT(numberFormat);

    // 4. Return InitializeNumberFormat(numberFormat, locales, options).
    scope.release();
    numberFormat->initializeNumberFormat(globalObject, callFrame->argument(0), callFrame->argument(1));
    return JSValue::encode(numberFormat);
}

JSC_DEFINE_HOST_FUNCTION(callIntlNumberFormat, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    // 11.1.2 Intl.NumberFormat ([locales [, options]]) (ECMA-402 2.0)
    // 1. If NewTarget is undefined, let newTarget be the active function object, else let newTarget be NewTarget.
    // NewTarget is always undefined when called as a function.

    // FIXME: Workaround to provide compatibility with ECMA-402 1.0 call/apply patterns.
    // https://bugs.webkit.org/show_bug.cgi?id=153679
    return JSValue::encode(constructIntlInstanceWithWorkaroundForLegacyIntlConstructor(globalObject, callFrame->thisValue(), callFrame->jsCallee(), [&] (VM& vm) {
        // 2. Let numberFormat be OrdinaryCreateFromConstructor(newTarget, %NumberFormatPrototype%).
        // 3. ReturnIfAbrupt(numberFormat).
        IntlNumberFormat* numberFormat = IntlNumberFormat::create(vm, globalObject->numberFormatStructure());
        ASSERT(numberFormat);

        // 4. Return InitializeNumberFormat(numberFormat, locales, options).
        numberFormat->initializeNumberFormat(globalObject, callFrame->argument(0), callFrame->argument(1));
        return numberFormat;
    }));
}

JSC_DEFINE_HOST_FUNCTION(intlNumberFormatConstructorFuncSupportedLocalesOf, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    // 11.2.2 Intl.NumberFormat.supportedLocalesOf(locales [, options]) (ECMA-402 2.0)

    // 1. Let availableLocales be %NumberFormat%.[[availableLocales]].
    const auto& availableLocales = intlNumberFormatAvailableLocales();

    // 2. Let requestedLocales be CanonicalizeLocaleList(locales).
    Vector<String> requestedLocales = canonicalizeLocaleList(globalObject, callFrame->argument(0));
    RETURN_IF_EXCEPTION(scope, encodedJSValue());

    // 3. Return SupportedLocales(availableLocales, requestedLocales, options).
    RELEASE_AND_RETURN(scope, JSValue::encode(supportedLocales(globalObject, availableLocales, requestedLocales, callFrame->argument(1))));
}

} // namespace JSC
