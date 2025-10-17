/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 29, 2023.
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
#include "IntlDurationFormatConstructor.h"

#include "IntlDurationFormat.h"
#include "IntlDurationFormatPrototype.h"
#include "JSCInlines.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(IntlDurationFormatConstructor);

static JSC_DECLARE_HOST_FUNCTION(intlDurationFormatConstructorSupportedLocalesOf);

}

#include "IntlDurationFormatConstructor.lut.h"

namespace JSC {

const ClassInfo IntlDurationFormatConstructor::s_info = { "Function"_s, &Base::s_info, &durationFormatConstructorTable, nullptr, CREATE_METHOD_TABLE(IntlDurationFormatConstructor) };

/* Source for IntlDurationFormatConstructor.lut.h
@begin durationFormatConstructorTable
  supportedLocalesOf             intlDurationFormatConstructorSupportedLocalesOf             DontEnum|Function 1
@end
*/

IntlDurationFormatConstructor* IntlDurationFormatConstructor::create(VM& vm, Structure* structure, IntlDurationFormatPrototype* durationFormatPrototype)
{
    auto* constructor = new (NotNull, allocateCell<IntlDurationFormatConstructor>(vm)) IntlDurationFormatConstructor(vm, structure);
    constructor->finishCreation(vm, durationFormatPrototype);
    return constructor;
}

Structure* IntlDurationFormatConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

static JSC_DECLARE_HOST_FUNCTION(callIntlDurationFormat);
static JSC_DECLARE_HOST_FUNCTION(constructIntlDurationFormat);

IntlDurationFormatConstructor::IntlDurationFormatConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, callIntlDurationFormat, constructIntlDurationFormat)
{
}

void IntlDurationFormatConstructor::finishCreation(VM& vm, IntlDurationFormatPrototype* durationFormatPrototype)
{
    Base::finishCreation(vm, 0, "DurationFormat"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, durationFormatPrototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
    durationFormatPrototype->putDirectWithoutTransition(vm, vm.propertyNames->constructor, this, static_cast<unsigned>(PropertyAttribute::DontEnum));
}

// https://tc39.es/proposal-intl-duration-format/#sec-Intl.DurationFormat
JSC_DEFINE_HOST_FUNCTION(constructIntlDurationFormat, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* structure = JSC_GET_DERIVED_STRUCTURE(vm, durationFormatStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    IntlDurationFormat* durationFormat = IntlDurationFormat::create(vm, structure);
    ASSERT(durationFormat);

    scope.release();
    durationFormat->initializeDurationFormat(globalObject, callFrame->argument(0), callFrame->argument(1));
    return JSValue::encode(durationFormat);
}

// https://tc39.es/proposal-intl-duration-format/#sec-Intl.DurationFormat
JSC_DEFINE_HOST_FUNCTION(callIntlDurationFormat, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    return JSValue::encode(throwConstructorCannotBeCalledAsFunctionTypeError(globalObject, scope, "DurationFormat"_s));
}

JSC_DEFINE_HOST_FUNCTION(intlDurationFormatConstructorSupportedLocalesOf, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    // Intl.DurationFormat.supportedLocalesOf(locales [, options])
    // https://tc39.es/proposal-intl-duration-format/#sec-Intl.DurationFormat.supportedLocalesOf

    // 1. Let availableLocales be %DurationFormat%.[[availableLocales]].
    const auto& availableLocales = intlDurationFormatAvailableLocales();

    // 2. Let requestedLocales be CanonicalizeLocaleList(locales).
    Vector<String> requestedLocales = canonicalizeLocaleList(globalObject, callFrame->argument(0));
    RETURN_IF_EXCEPTION(scope, encodedJSValue());

    // 3. Return SupportedLocales(availableLocales, requestedLocales, options).
    RELEASE_AND_RETURN(scope, JSValue::encode(supportedLocales(globalObject, availableLocales, requestedLocales, callFrame->argument(1))));
}

} // namespace JSC
