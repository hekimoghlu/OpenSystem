/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 10, 2021.
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
#include "IntlRelativeTimeFormatConstructor.h"

#include "IntlObject.h"
#include "IntlRelativeTimeFormat.h"
#include "IntlRelativeTimeFormatPrototype.h"
#include "JSCInlines.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(IntlRelativeTimeFormatConstructor);

static JSC_DECLARE_HOST_FUNCTION(intlRelativeTimeFormatConstructorFuncSupportedLocalesOf);

}

#include "IntlRelativeTimeFormatConstructor.lut.h"

namespace JSC {

const ClassInfo IntlRelativeTimeFormatConstructor::s_info = { "Function"_s, &InternalFunction::s_info, &relativeTimeFormatConstructorTable, nullptr, CREATE_METHOD_TABLE(IntlRelativeTimeFormatConstructor) };

/* Source for IntlRelativeTimeFormatConstructor.lut.h
@begin relativeTimeFormatConstructorTable
  supportedLocalesOf             intlRelativeTimeFormatConstructorFuncSupportedLocalesOf             DontEnum|Function 1
@end
*/

IntlRelativeTimeFormatConstructor* IntlRelativeTimeFormatConstructor::create(VM& vm, Structure* structure, IntlRelativeTimeFormatPrototype* relativeTimeFormatPrototype)
{
    auto* constructor = new (NotNull, allocateCell<IntlRelativeTimeFormatConstructor>(vm)) IntlRelativeTimeFormatConstructor(vm, structure);
    constructor->finishCreation(vm, relativeTimeFormatPrototype);
    return constructor;
}

Structure* IntlRelativeTimeFormatConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

static JSC_DECLARE_HOST_FUNCTION(callIntlRelativeTimeFormat);
static JSC_DECLARE_HOST_FUNCTION(constructIntlRelativeTimeFormat);

IntlRelativeTimeFormatConstructor::IntlRelativeTimeFormatConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, callIntlRelativeTimeFormat, constructIntlRelativeTimeFormat)
{
}

void IntlRelativeTimeFormatConstructor::finishCreation(VM& vm, IntlRelativeTimeFormatPrototype* relativeTimeFormatPrototype)
{
    Base::finishCreation(vm, 0, "RelativeTimeFormat"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, relativeTimeFormatPrototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
    relativeTimeFormatPrototype->putDirectWithoutTransition(vm, vm.propertyNames->constructor, this, static_cast<unsigned>(PropertyAttribute::DontEnum));
}

// https://tc39.es/ecma402/#sec-Intl.RelativeTimeFormat
JSC_DEFINE_HOST_FUNCTION(constructIntlRelativeTimeFormat, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* structure = JSC_GET_DERIVED_STRUCTURE(vm, relativeTimeFormatStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    IntlRelativeTimeFormat* relativeTimeFormat = IntlRelativeTimeFormat::create(vm, structure);
    ASSERT(relativeTimeFormat);

    scope.release();
    relativeTimeFormat->initializeRelativeTimeFormat(globalObject, callFrame->argument(0), callFrame->argument(1));
    return JSValue::encode(relativeTimeFormat);
}

// https://tc39.es/ecma402/#sec-Intl.RelativeTimeFormat
JSC_DEFINE_HOST_FUNCTION(callIntlRelativeTimeFormat, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    return JSValue::encode(throwConstructorCannotBeCalledAsFunctionTypeError(globalObject, scope, "RelativeTimeFormat"_s));
}

// https://tc39.es/ecma402/#sec-Intl.RelativeTimeFormat.supportedLocalesOf
JSC_DEFINE_HOST_FUNCTION(intlRelativeTimeFormatConstructorFuncSupportedLocalesOf, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    const auto& availableLocales = intlRelativeTimeFormatAvailableLocales();

    auto requestedLocales = canonicalizeLocaleList(globalObject, callFrame->argument(0));
    RETURN_IF_EXCEPTION(scope, encodedJSValue());

    RELEASE_AND_RETURN(scope, JSValue::encode(supportedLocales(globalObject, availableLocales, requestedLocales, callFrame->argument(1))));
}

} // namespace JSC
