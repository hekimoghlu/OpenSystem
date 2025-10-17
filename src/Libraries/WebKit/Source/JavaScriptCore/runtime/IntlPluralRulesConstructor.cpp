/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 11, 2025.
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
#include "IntlPluralRulesConstructor.h"

#include "IntlObject.h"
#include "IntlPluralRules.h"
#include "IntlPluralRulesPrototype.h"
#include "JSCInlines.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(IntlPluralRulesConstructor);

static JSC_DECLARE_HOST_FUNCTION(intlPluralRulesConstructorFuncSupportedLocalesOf);

}

#include "IntlPluralRulesConstructor.lut.h"

namespace JSC {

const ClassInfo IntlPluralRulesConstructor::s_info = { "Function"_s, &InternalFunction::s_info, &pluralRulesConstructorTable, nullptr, CREATE_METHOD_TABLE(IntlPluralRulesConstructor) };

/* Source for IntlPluralRulesConstructor.lut.h
@begin pluralRulesConstructorTable
  supportedLocalesOf             intlPluralRulesConstructorFuncSupportedLocalesOf             DontEnum|Function 1
@end
*/

IntlPluralRulesConstructor* IntlPluralRulesConstructor::create(VM& vm, Structure* structure, IntlPluralRulesPrototype* pluralRulesPrototype)
{
    IntlPluralRulesConstructor* constructor = new (NotNull, allocateCell<IntlPluralRulesConstructor>(vm)) IntlPluralRulesConstructor(vm, structure);
    constructor->finishCreation(vm, pluralRulesPrototype);
    return constructor;
}

Structure* IntlPluralRulesConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

static JSC_DECLARE_HOST_FUNCTION(callIntlPluralRules);
static JSC_DECLARE_HOST_FUNCTION(constructIntlPluralRules);

IntlPluralRulesConstructor::IntlPluralRulesConstructor(VM& vm, Structure* structure)
    : InternalFunction(vm, structure, callIntlPluralRules, constructIntlPluralRules)
{
}

void IntlPluralRulesConstructor::finishCreation(VM& vm, IntlPluralRulesPrototype* pluralRulesPrototype)
{
    Base::finishCreation(vm, 0, "PluralRules"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, pluralRulesPrototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
    pluralRulesPrototype->putDirectWithoutTransition(vm, vm.propertyNames->constructor, this, static_cast<unsigned>(PropertyAttribute::DontEnum));
}

JSC_DEFINE_HOST_FUNCTION(constructIntlPluralRules, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    // 13.2.1 Intl.PluralRules ([ locales [ , options ] ])
    // https://tc39.github.io/ecma402/#sec-intl.pluralrules
    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* structure = JSC_GET_DERIVED_STRUCTURE(vm, pluralRulesStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    IntlPluralRules* pluralRules = IntlPluralRules::create(vm, structure);
    ASSERT(pluralRules);

    scope.release();
    pluralRules->initializePluralRules(globalObject, callFrame->argument(0), callFrame->argument(1));
    return JSValue::encode(pluralRules);
}

JSC_DEFINE_HOST_FUNCTION(callIntlPluralRules, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    // 13.2.1 Intl.PluralRules ([ locales [ , options ] ])
    // https://tc39.github.io/ecma402/#sec-intl.pluralrules
    return JSValue::encode(throwConstructorCannotBeCalledAsFunctionTypeError(globalObject, scope, "PluralRules"_s));
}

JSC_DEFINE_HOST_FUNCTION(intlPluralRulesConstructorFuncSupportedLocalesOf, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    // 13.3.2 Intl.PluralRules.supportedLocalesOf (locales [, options ])
    // https://tc39.github.io/ecma402/#sec-intl.pluralrules.supportedlocalesof
    const auto& availableLocales = intlPluralRulesAvailableLocales();

    Vector<String> requestedLocales = canonicalizeLocaleList(globalObject, callFrame->argument(0));
    RETURN_IF_EXCEPTION(scope, encodedJSValue());

    RELEASE_AND_RETURN(scope, JSValue::encode(supportedLocales(globalObject, availableLocales, requestedLocales, callFrame->argument(1))));
}

} // namespace JSC
