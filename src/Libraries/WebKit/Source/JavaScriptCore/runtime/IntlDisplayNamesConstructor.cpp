/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 28, 2025.
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
#include "IntlDisplayNamesConstructor.h"

#include "IntlDisplayNames.h"
#include "IntlDisplayNamesPrototype.h"
#include "JSCInlines.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(IntlDisplayNamesConstructor);

static JSC_DECLARE_HOST_FUNCTION(intlDisplayNamesConstructorSupportedLocalesOf);

}

#include "IntlDisplayNamesConstructor.lut.h"

namespace JSC {

const ClassInfo IntlDisplayNamesConstructor::s_info = { "Function"_s, &Base::s_info, &displayNamesConstructorTable, nullptr, CREATE_METHOD_TABLE(IntlDisplayNamesConstructor) };

/* Source for IntlDisplayNamesConstructor.lut.h
@begin displayNamesConstructorTable
  supportedLocalesOf             intlDisplayNamesConstructorSupportedLocalesOf             DontEnum|Function 1
@end
*/

IntlDisplayNamesConstructor* IntlDisplayNamesConstructor::create(VM& vm, Structure* structure, IntlDisplayNamesPrototype* displayNamesPrototype)
{
    auto* constructor = new (NotNull, allocateCell<IntlDisplayNamesConstructor>(vm)) IntlDisplayNamesConstructor(vm, structure);
    constructor->finishCreation(vm, displayNamesPrototype);
    return constructor;
}

Structure* IntlDisplayNamesConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

static JSC_DECLARE_HOST_FUNCTION(callIntlDisplayNames);
static JSC_DECLARE_HOST_FUNCTION(constructIntlDisplayNames);

IntlDisplayNamesConstructor::IntlDisplayNamesConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, callIntlDisplayNames, constructIntlDisplayNames)
{
}

void IntlDisplayNamesConstructor::finishCreation(VM& vm, IntlDisplayNamesPrototype* displayNamesPrototype)
{
    Base::finishCreation(vm, 2, "DisplayNames"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, displayNamesPrototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
    displayNamesPrototype->putDirectWithoutTransition(vm, vm.propertyNames->constructor, this, static_cast<unsigned>(PropertyAttribute::DontEnum));
}

// https://tc39.es/ecma402/#sec-Intl.DisplayNames
JSC_DEFINE_HOST_FUNCTION(constructIntlDisplayNames, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* structure = JSC_GET_DERIVED_STRUCTURE(vm, displayNamesStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    IntlDisplayNames* displayNames = IntlDisplayNames::create(vm, structure);
    ASSERT(displayNames);

    scope.release();
    displayNames->initializeDisplayNames(globalObject, callFrame->argument(0), callFrame->argument(1));
    return JSValue::encode(displayNames);
}

// https://tc39.es/ecma402/#sec-Intl.DisplayNames
JSC_DEFINE_HOST_FUNCTION(callIntlDisplayNames, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    return JSValue::encode(throwConstructorCannotBeCalledAsFunctionTypeError(globalObject, scope, "DisplayNames"_s));
}

JSC_DEFINE_HOST_FUNCTION(intlDisplayNamesConstructorSupportedLocalesOf, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    // Intl.DisplayNames.supportedLocalesOf(locales [, options]) (ECMA-402 2.0)
    // https://tc39.es/proposal-intl-displaynames/#sec-Intl.DisplayNames.supportedLocalesOf

    // 1. Let availableLocales be %DisplayNames%.[[availableLocales]].
    const auto& availableLocales = intlDisplayNamesAvailableLocales();

    // 2. Let requestedLocales be CanonicalizeLocaleList(locales).
    Vector<String> requestedLocales = canonicalizeLocaleList(globalObject, callFrame->argument(0));
    RETURN_IF_EXCEPTION(scope, encodedJSValue());

    // 3. Return SupportedLocales(availableLocales, requestedLocales, options).
    RELEASE_AND_RETURN(scope, JSValue::encode(supportedLocales(globalObject, availableLocales, requestedLocales, callFrame->argument(1))));
}


} // namespace JSC
