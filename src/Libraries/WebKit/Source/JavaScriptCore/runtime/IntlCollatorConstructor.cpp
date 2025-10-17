/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 28, 2023.
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
#include "IntlCollatorConstructor.h"

#include "IntlCollator.h"
#include "IntlCollatorPrototype.h"
#include "IntlObject.h"
#include "JSCInlines.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(IntlCollatorConstructor);

static JSC_DECLARE_HOST_FUNCTION(intlCollatorConstructorFuncSupportedLocalesOf);

}

#include "IntlCollatorConstructor.lut.h"

namespace JSC {

static JSC_DECLARE_HOST_FUNCTION(callIntlCollator);
static JSC_DECLARE_HOST_FUNCTION(constructIntlCollator);

const ClassInfo IntlCollatorConstructor::s_info = { "Function"_s, &InternalFunction::s_info, &collatorConstructorTable, nullptr, CREATE_METHOD_TABLE(IntlCollatorConstructor) };

/* Source for IntlCollatorConstructor.lut.h
@begin collatorConstructorTable
  supportedLocalesOf             intlCollatorConstructorFuncSupportedLocalesOf             DontEnum|Function 1
@end
*/

IntlCollatorConstructor* IntlCollatorConstructor::create(VM& vm, Structure* structure, IntlCollatorPrototype* collatorPrototype)
{
    IntlCollatorConstructor* constructor = new (NotNull, allocateCell<IntlCollatorConstructor>(vm)) IntlCollatorConstructor(vm, structure);
    constructor->finishCreation(vm, collatorPrototype);
    return constructor;
}

Structure* IntlCollatorConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

IntlCollatorConstructor::IntlCollatorConstructor(VM& vm, Structure* structure)
    : InternalFunction(vm, structure, callIntlCollator, constructIntlCollator)
{
}

void IntlCollatorConstructor::finishCreation(VM& vm, IntlCollatorPrototype* collatorPrototype)
{
    Base::finishCreation(vm, 0, "Collator"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, collatorPrototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
    collatorPrototype->putDirectWithoutTransition(vm, vm.propertyNames->constructor, this, static_cast<unsigned>(PropertyAttribute::DontEnum));
}

JSC_DEFINE_HOST_FUNCTION(constructIntlCollator, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    // 10.1.2 Intl.Collator ([locales [, options]]) (ECMA-402 2.0)
    // 1. If NewTarget is undefined, let newTarget be the active function object, else let newTarget be NewTarget.
    // 2. Let collator be OrdinaryCreateFromConstructor(newTarget, %CollatorPrototype%).
    // 3. ReturnIfAbrupt(collator).
    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* structure = JSC_GET_DERIVED_STRUCTURE(vm, collatorStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    IntlCollator* collator = IntlCollator::create(vm, structure);
    ASSERT(collator);

    // 4. Return InitializeCollator(collator, locales, options).
    scope.release();
    collator->initializeCollator(globalObject, callFrame->argument(0), callFrame->argument(1));
    return JSValue::encode(collator);
}

JSC_DEFINE_HOST_FUNCTION(callIntlCollator, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    // 10.1.2 Intl.Collator ([locales [, options]]) (ECMA-402 2.0)
    // 1. If NewTarget is undefined, let newTarget be the active function object, else let newTarget be NewTarget.
    // NewTarget is always undefined when called as a function.

    VM& vm = globalObject->vm();
    // Collator does not require the workaround for ECMA-402 1.0 compatibility.
    // https://bugs.webkit.org/show_bug.cgi?id=153679

    // 2. Let collator be OrdinaryCreateFromConstructor(newTarget, %CollatorPrototype%).
    // 3. ReturnIfAbrupt(collator).
    IntlCollator* collator = IntlCollator::create(vm, globalObject->collatorStructure());
    ASSERT(collator);

    // 4. Return InitializeCollator(collator, locales, options).
    collator->initializeCollator(globalObject, callFrame->argument(0), callFrame->argument(1));
    return JSValue::encode(collator);
}

JSC_DEFINE_HOST_FUNCTION(intlCollatorConstructorFuncSupportedLocalesOf, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    // 10.2.2 Intl.Collator.supportedLocalesOf(locales [, options]) (ECMA-402 2.0)

    // 1. Let requestedLocales be CanonicalizeLocaleList(locales).
    Vector<String> requestedLocales = canonicalizeLocaleList(globalObject, callFrame->argument(0));

    // 2. ReturnIfAbrupt(requestedLocales).
    RETURN_IF_EXCEPTION(scope, encodedJSValue());

    // 3. Return SupportedLocales(%Collator%.[[availableLocales]], requestedLocales, options).
    RELEASE_AND_RETURN(scope, JSValue::encode(supportedLocales(globalObject, intlCollatorAvailableLocales(), requestedLocales, callFrame->argument(1))));
}

} // namespace JSC
