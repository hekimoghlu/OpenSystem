/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 6, 2024.
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
#include "IntlLocaleConstructor.h"

#include "IntlLocale.h"
#include "IntlLocalePrototype.h"
#include "JSCInlines.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(IntlLocaleConstructor);

const ClassInfo IntlLocaleConstructor::s_info = { "Function"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(IntlLocaleConstructor) };

IntlLocaleConstructor* IntlLocaleConstructor::create(VM& vm, Structure* structure, IntlLocalePrototype* localePrototype)
{
    auto* constructor = new (NotNull, allocateCell<IntlLocaleConstructor>(vm)) IntlLocaleConstructor(vm, structure);
    constructor->finishCreation(vm, localePrototype);
    return constructor;
}

Structure* IntlLocaleConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

static JSC_DECLARE_HOST_FUNCTION(callIntlLocale);
static JSC_DECLARE_HOST_FUNCTION(constructIntlLocale);

IntlLocaleConstructor::IntlLocaleConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, callIntlLocale, constructIntlLocale)
{
}

void IntlLocaleConstructor::finishCreation(VM& vm, IntlLocalePrototype* localePrototype)
{
    Base::finishCreation(vm, 1, "Locale"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, localePrototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
    localePrototype->putDirectWithoutTransition(vm, vm.propertyNames->constructor, this, static_cast<unsigned>(PropertyAttribute::DontEnum));
}

// https://tc39.es/ecma402/#sec-Intl.Locale
JSC_DEFINE_HOST_FUNCTION(constructIntlLocale, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* structure = JSC_GET_DERIVED_STRUCTURE(vm, localeStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    IntlLocale* locale = IntlLocale::create(vm, structure);
    ASSERT(locale);

    JSValue tag = callFrame->argument(0);
    if (!tag.isString() && !tag.isObject())
        return throwVMTypeError(globalObject, scope, "First argument to Intl.Locale must be a string or an object"_s);

    scope.release();
    locale->initializeLocale(globalObject, tag, callFrame->argument(1));
    return JSValue::encode(locale);
}

// https://tc39.es/ecma402/#sec-Intl.Locale
JSC_DEFINE_HOST_FUNCTION(callIntlLocale, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    return JSValue::encode(throwConstructorCannotBeCalledAsFunctionTypeError(globalObject, scope, "Locale"_s));
}

} // namespace JSC
