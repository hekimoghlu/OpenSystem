/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 20, 2024.
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
#include "IntlSegmenterConstructor.h"

#include "IntlSegmenter.h"
#include "IntlSegmenterPrototype.h"
#include "JSCInlines.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(IntlSegmenterConstructor);

static JSC_DECLARE_HOST_FUNCTION(intlSegmenterConstructorSupportedLocalesOf);

}

#include "IntlSegmenterConstructor.lut.h"

namespace JSC {

const ClassInfo IntlSegmenterConstructor::s_info = { "Function"_s, &Base::s_info, &segmenterConstructorTable, nullptr, CREATE_METHOD_TABLE(IntlSegmenterConstructor) };

/* Source for IntlSegmenterConstructor.lut.h
@begin segmenterConstructorTable
  supportedLocalesOf             intlSegmenterConstructorSupportedLocalesOf             DontEnum|Function 1
@end
*/

IntlSegmenterConstructor* IntlSegmenterConstructor::create(VM& vm, Structure* structure, IntlSegmenterPrototype* segmenterPrototype)
{
    auto* constructor = new (NotNull, allocateCell<IntlSegmenterConstructor>(vm)) IntlSegmenterConstructor(vm, structure);
    constructor->finishCreation(vm, segmenterPrototype);
    return constructor;
}

Structure* IntlSegmenterConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

static JSC_DECLARE_HOST_FUNCTION(callIntlSegmenter);
static JSC_DECLARE_HOST_FUNCTION(constructIntlSegmenter);

IntlSegmenterConstructor::IntlSegmenterConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, callIntlSegmenter, constructIntlSegmenter)
{
}

void IntlSegmenterConstructor::finishCreation(VM& vm, IntlSegmenterPrototype* segmenterPrototype)
{
    Base::finishCreation(vm, 0, "Segmenter"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, segmenterPrototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
    segmenterPrototype->putDirectWithoutTransition(vm, vm.propertyNames->constructor, this, static_cast<unsigned>(PropertyAttribute::DontEnum));
}

// https://tc39.es/ecma402/#sec-Intl.Segmenter
JSC_DEFINE_HOST_FUNCTION(constructIntlSegmenter, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* structure = JSC_GET_DERIVED_STRUCTURE(vm, segmenterStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    IntlSegmenter* segmenter = IntlSegmenter::create(vm, structure);
    ASSERT(segmenter);

    scope.release();
    segmenter->initializeSegmenter(globalObject, callFrame->argument(0), callFrame->argument(1));
    return JSValue::encode(segmenter);
}

// https://tc39.es/ecma402/#sec-Intl.Segmenter
JSC_DEFINE_HOST_FUNCTION(callIntlSegmenter, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    return JSValue::encode(throwConstructorCannotBeCalledAsFunctionTypeError(globalObject, scope, "Segmenter"_s));
}

JSC_DEFINE_HOST_FUNCTION(intlSegmenterConstructorSupportedLocalesOf, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    // https://tc39.es/proposal-intl-segmenter/#sec-intl.segmenter.supportedlocalesof

    // 1. Let availableLocales be %Segmenter%.[[availableLocales]].
    const auto& availableLocales = intlSegmenterAvailableLocales();

    // 2. Let requestedLocales be CanonicalizeLocaleList(locales).
    Vector<String> requestedLocales = canonicalizeLocaleList(globalObject, callFrame->argument(0));
    RETURN_IF_EXCEPTION(scope, encodedJSValue());

    // 3. Return SupportedLocales(availableLocales, requestedLocales, options).
    RELEASE_AND_RETURN(scope, JSValue::encode(supportedLocales(globalObject, availableLocales, requestedLocales, callFrame->argument(1))));
}


} // namespace JSC
