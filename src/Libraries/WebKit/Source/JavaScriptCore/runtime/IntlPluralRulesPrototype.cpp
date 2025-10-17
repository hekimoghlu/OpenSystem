/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 6, 2024.
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
#include "IntlPluralRulesPrototype.h"

#include "IntlPluralRules.h"
#include "JSCInlines.h"

namespace JSC {

static JSC_DECLARE_HOST_FUNCTION(intlPluralRulesPrototypeFuncSelect);
static JSC_DECLARE_HOST_FUNCTION(intlPluralRulesPrototypeFuncSelectRange);
static JSC_DECLARE_HOST_FUNCTION(intlPluralRulesPrototypeFuncResolvedOptions);

}

#include "IntlPluralRulesPrototype.lut.h"

namespace JSC {

const ClassInfo IntlPluralRulesPrototype::s_info = { "Intl.PluralRules"_s, &Base::s_info, &pluralRulesPrototypeTable, nullptr, CREATE_METHOD_TABLE(IntlPluralRulesPrototype) };

/* Source for IntlPluralRulesPrototype.lut.h
@begin pluralRulesPrototypeTable
  select           intlPluralRulesPrototypeFuncSelect           DontEnum|Function 1
  selectRange      intlPluralRulesPrototypeFuncSelectRange      DontEnum|Function 2
  resolvedOptions  intlPluralRulesPrototypeFuncResolvedOptions  DontEnum|Function 0
@end
*/

IntlPluralRulesPrototype* IntlPluralRulesPrototype::create(VM& vm, JSGlobalObject* globalObject, Structure* structure)
{
    IntlPluralRulesPrototype* object = new (NotNull, allocateCell<IntlPluralRulesPrototype>(vm)) IntlPluralRulesPrototype(vm, structure);
    object->finishCreation(vm, globalObject);
    return object;
}

Structure* IntlPluralRulesPrototype::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
}

IntlPluralRulesPrototype::IntlPluralRulesPrototype(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

void IntlPluralRulesPrototype::finishCreation(VM& vm, JSGlobalObject* globalObject)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
    UNUSED_PARAM(globalObject);
}

JSC_DEFINE_HOST_FUNCTION(intlPluralRulesPrototypeFuncSelect, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    // 13.4.3 Intl.PluralRules.prototype.select (value)
    // https://tc39.github.io/ecma402/#sec-intl.pluralrules.prototype.select
    IntlPluralRules* pluralRules = jsDynamicCast<IntlPluralRules*>(callFrame->thisValue());

    if (UNLIKELY(!pluralRules))
        return JSValue::encode(throwTypeError(globalObject, scope, "Intl.PluralRules.prototype.select called on value that's not a PluralRules"_s));

    double value = callFrame->argument(0).toNumber(globalObject);
    RETURN_IF_EXCEPTION(scope, encodedJSValue());

    RELEASE_AND_RETURN(scope, JSValue::encode(pluralRules->select(globalObject, value)));
}

JSC_DEFINE_HOST_FUNCTION(intlPluralRulesPrototypeFuncSelectRange, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    // https://tc39.es/proposal-intl-numberformat-v3/out/pluralrules/diff.html#sec-intl.pluralrules.prototype.selectrange
    IntlPluralRules* pluralRules = jsDynamicCast<IntlPluralRules*>(callFrame->thisValue());
    if (UNLIKELY(!pluralRules))
        return JSValue::encode(throwTypeError(globalObject, scope, "Intl.PluralRules.prototype.selectRange called on value that's not a PluralRules"_s));

    JSValue startValue = callFrame->argument(0);
    JSValue endValue = callFrame->argument(1);

    if (startValue.isUndefined() || endValue.isUndefined())
        return throwVMTypeError(globalObject, scope, "start or end is undefined"_s);

    double start = startValue.toNumber(globalObject);
    RETURN_IF_EXCEPTION(scope, { });

    double end = endValue.toNumber(globalObject);
    RETURN_IF_EXCEPTION(scope, { });

    RELEASE_AND_RETURN(scope, JSValue::encode(pluralRules->selectRange(globalObject, start, end)));
}

JSC_DEFINE_HOST_FUNCTION(intlPluralRulesPrototypeFuncResolvedOptions, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    // 13.4.4 Intl.PluralRules.prototype.resolvedOptions ()
    // https://tc39.github.io/ecma402/#sec-intl.pluralrules.prototype.resolvedoptions
    IntlPluralRules* pluralRules = jsDynamicCast<IntlPluralRules*>(callFrame->thisValue());

    if (UNLIKELY(!pluralRules))
        return JSValue::encode(throwTypeError(globalObject, scope, "Intl.PluralRules.prototype.resolvedOptions called on value that's not a PluralRules"_s));

    RELEASE_AND_RETURN(scope, JSValue::encode(pluralRules->resolvedOptions(globalObject)));
}

} // namespace JSC
