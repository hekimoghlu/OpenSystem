/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 28, 2023.
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
#include "IntlListFormatConstructor.h"

#include "IntlListFormat.h"
#include "IntlListFormatPrototype.h"
#include "JSCInlines.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(IntlListFormatConstructor);

static JSC_DECLARE_HOST_FUNCTION(intlListFormatConstructorSupportedLocalesOf);

}

#include "IntlListFormatConstructor.lut.h"

namespace JSC {

const ClassInfo IntlListFormatConstructor::s_info = { "Function"_s, &Base::s_info, &listFormatConstructorTable, nullptr, CREATE_METHOD_TABLE(IntlListFormatConstructor) };

/* Source for IntlListFormatConstructor.lut.h
@begin listFormatConstructorTable
  supportedLocalesOf             intlListFormatConstructorSupportedLocalesOf             DontEnum|Function 1
@end
*/

IntlListFormatConstructor* IntlListFormatConstructor::create(VM& vm, Structure* structure, IntlListFormatPrototype* listFormatPrototype)
{
    auto* constructor = new (NotNull, allocateCell<IntlListFormatConstructor>(vm)) IntlListFormatConstructor(vm, structure);
    constructor->finishCreation(vm, listFormatPrototype);
    return constructor;
}

Structure* IntlListFormatConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

static JSC_DECLARE_HOST_FUNCTION(callIntlListFormat);
static JSC_DECLARE_HOST_FUNCTION(constructIntlListFormat);

IntlListFormatConstructor::IntlListFormatConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, callIntlListFormat, constructIntlListFormat)
{
}

void IntlListFormatConstructor::finishCreation(VM& vm, IntlListFormatPrototype* listFormatPrototype)
{
    Base::finishCreation(vm, 0, "ListFormat"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, listFormatPrototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
    listFormatPrototype->putDirectWithoutTransition(vm, vm.propertyNames->constructor, this, static_cast<unsigned>(PropertyAttribute::DontEnum));
}

// https://tc39.es/proposal-intl-list-format/#sec-Intl.ListFormat
JSC_DEFINE_HOST_FUNCTION(constructIntlListFormat, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* structure = JSC_GET_DERIVED_STRUCTURE(vm, listFormatStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    IntlListFormat* listFormat = IntlListFormat::create(vm, structure);
    ASSERT(listFormat);

    scope.release();
    listFormat->initializeListFormat(globalObject, callFrame->argument(0), callFrame->argument(1));
    return JSValue::encode(listFormat);
}

// https://tc39.es/proposal-intl-list-format/#sec-Intl.ListFormat
JSC_DEFINE_HOST_FUNCTION(callIntlListFormat, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    return JSValue::encode(throwConstructorCannotBeCalledAsFunctionTypeError(globalObject, scope, "ListFormat"_s));
}

JSC_DEFINE_HOST_FUNCTION(intlListFormatConstructorSupportedLocalesOf, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    // Intl.ListFormat.supportedLocalesOf(locales [, options])
    // https://tc39.es/proposal-intl-list-format/#sec-Intl.ListFormat.supportedLocalesOf

    // 1. Let availableLocales be %ListFormat%.[[availableLocales]].
    const auto& availableLocales = intlListFormatAvailableLocales();

    // 2. Let requestedLocales be CanonicalizeLocaleList(locales).
    Vector<String> requestedLocales = canonicalizeLocaleList(globalObject, callFrame->argument(0));
    RETURN_IF_EXCEPTION(scope, encodedJSValue());

    // 3. Return SupportedLocales(availableLocales, requestedLocales, options).
    RELEASE_AND_RETURN(scope, JSValue::encode(supportedLocales(globalObject, availableLocales, requestedLocales, callFrame->argument(1))));
}


} // namespace JSC
