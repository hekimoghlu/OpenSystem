/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 22, 2021.
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
#include "IntlNumberFormatPrototype.h"

#include "BuiltinNames.h"
#include "IntlNumberFormatInlines.h"
#include "JSBoundFunction.h"
#include "JSCInlines.h"

namespace JSC {

static JSC_DECLARE_CUSTOM_GETTER(intlNumberFormatPrototypeGetterFormat);
static JSC_DECLARE_HOST_FUNCTION(intlNumberFormatPrototypeFuncFormatRange);
static JSC_DECLARE_HOST_FUNCTION(intlNumberFormatPrototypeFuncFormatRangeToParts);
static JSC_DECLARE_HOST_FUNCTION(intlNumberFormatPrototypeFuncFormatToParts);
static JSC_DECLARE_HOST_FUNCTION(intlNumberFormatPrototypeFuncResolvedOptions);
static JSC_DECLARE_HOST_FUNCTION(intlNumberFormatFuncFormat);

}

#include "IntlNumberFormatPrototype.lut.h"

namespace JSC {

const ClassInfo IntlNumberFormatPrototype::s_info = { "Intl.NumberFormat"_s, &Base::s_info, &numberFormatPrototypeTable, nullptr, CREATE_METHOD_TABLE(IntlNumberFormatPrototype) };

/* Source for IntlNumberFormatPrototype.lut.h
@begin numberFormatPrototypeTable
  format              intlNumberFormatPrototypeGetterFormat            DontEnum|ReadOnly|CustomAccessor
  formatRange         intlNumberFormatPrototypeFuncFormatRange         DontEnum|Function 2
  formatRangeToParts  intlNumberFormatPrototypeFuncFormatRangeToParts  DontEnum|Function 2
  formatToParts       intlNumberFormatPrototypeFuncFormatToParts       DontEnum|Function 1
  resolvedOptions     intlNumberFormatPrototypeFuncResolvedOptions     DontEnum|Function 0
@end
*/

IntlNumberFormatPrototype* IntlNumberFormatPrototype::create(VM& vm, JSGlobalObject* globalObject, Structure* structure)
{
    IntlNumberFormatPrototype* object = new (NotNull, allocateCell<IntlNumberFormatPrototype>(vm)) IntlNumberFormatPrototype(vm, structure);
    object->finishCreation(vm, globalObject);
    return object;
}

Structure* IntlNumberFormatPrototype::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
}

IntlNumberFormatPrototype::IntlNumberFormatPrototype(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

void IntlNumberFormatPrototype::finishCreation(VM& vm, JSGlobalObject* globalObject)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
    UNUSED_PARAM(globalObject);
}

// https://tc39.es/ecma402/#sec-number-format-functions
JSC_DEFINE_HOST_FUNCTION(intlNumberFormatFuncFormat, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    auto* numberFormat = jsDynamicCast<IntlNumberFormat*>(callFrame->thisValue());
    if (UNLIKELY(!numberFormat))
        return JSValue::encode(throwTypeError(globalObject, scope, "Intl.NumberFormat.prototype.format called on value that's not a NumberFormat"_s));

    auto value = toIntlMathematicalValue(globalObject, callFrame->argument(0));
    RETURN_IF_EXCEPTION(scope, { });

    if (auto number = value.tryGetDouble())
        RELEASE_AND_RETURN(scope, JSValue::encode(numberFormat->format(globalObject, number.value())));

    RELEASE_AND_RETURN(scope, JSValue::encode(numberFormat->format(globalObject, WTFMove(value))));
}

JSC_DEFINE_CUSTOM_GETTER(intlNumberFormatPrototypeGetterFormat, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    // 11.3.3 Intl.NumberFormat.prototype.format (ECMA-402 2.0)
    // 1. Let nf be this NumberFormat object.
    auto* nf = IntlNumberFormat::unwrapForOldFunctions(globalObject, JSValue::decode(thisValue));
    RETURN_IF_EXCEPTION(scope, { });
    if (UNLIKELY(!nf))
        return JSValue::encode(throwTypeError(globalObject, scope, "Intl.NumberFormat.prototype.format called on value that's not a NumberFormat"_s));

    JSBoundFunction* boundFormat = nf->boundFormat();
    // 2. If nf.[[boundFormat]] is undefined,
    if (!boundFormat) {
        JSGlobalObject* globalObject = nf->globalObject();
        // a. Let F be a new built-in function object as defined in 11.3.4.
        // b. The value of Fâ€™s length property is 1.
        auto* targetObject = JSFunction::create(vm, globalObject, 1, "format"_s, intlNumberFormatFuncFormat, ImplementationVisibility::Public);
        // c. Let bf be BoundFunctionCreate(F, Â«this valueÂ»).
        boundFormat = JSBoundFunction::create(vm, globalObject, targetObject, nf, { }, 1, jsEmptyString(vm));
        RETURN_IF_EXCEPTION(scope, { });
        boundFormat->reifyLazyPropertyIfNeeded<>(vm, globalObject, vm.propertyNames->name);
        RETURN_IF_EXCEPTION(scope, { });
        boundFormat->putDirect(vm, vm.propertyNames->name, jsEmptyString(vm), PropertyAttribute::ReadOnly | PropertyAttribute::DontEnum);
        // d. Set nf.[[boundFormat]] to bf.
        nf->setBoundFormat(vm, boundFormat);
    }
    // 3. Return nf.[[boundFormat]].
    return JSValue::encode(boundFormat);
}

JSC_DEFINE_HOST_FUNCTION(intlNumberFormatPrototypeFuncFormatRange, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    // Do not use unwrapForOldFunctions.
    auto* numberFormat = jsDynamicCast<IntlNumberFormat*>(callFrame->thisValue());
    if (UNLIKELY(!numberFormat))
        return JSValue::encode(throwTypeError(globalObject, scope, "Intl.NumberFormat.prototype.formatRange called on value that's not a NumberFormat"_s));

    JSValue startValue = callFrame->argument(0);
    JSValue endValue = callFrame->argument(1);

    if (startValue.isUndefined() || endValue.isUndefined())
        return throwVMTypeError(globalObject, scope, "start or end is undefined"_s);

    auto start = toIntlMathematicalValue(globalObject, startValue);
    RETURN_IF_EXCEPTION(scope, { });

    auto end = toIntlMathematicalValue(globalObject, endValue);
    RETURN_IF_EXCEPTION(scope, { });

    if (auto startNumber = start.tryGetDouble()) {
        if (auto endNumber = end.tryGetDouble())
            RELEASE_AND_RETURN(scope, JSValue::encode(numberFormat->formatRange(globalObject, startNumber.value(), endNumber.value())));
    }

    RELEASE_AND_RETURN(scope, JSValue::encode(numberFormat->formatRange(globalObject, WTFMove(start), WTFMove(end))));
}

JSC_DEFINE_HOST_FUNCTION(intlNumberFormatPrototypeFuncFormatToParts, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    // Intl.NumberFormat.prototype.formatToParts (ECMA-402)
    // https://tc39.github.io/ecma402/#sec-intl.numberformat.prototype.formattoparts

    // Do not use unwrapForOldFunctions.
    auto* numberFormat = jsDynamicCast<IntlNumberFormat*>(callFrame->thisValue());
    if (UNLIKELY(!numberFormat))
        return JSValue::encode(throwTypeError(globalObject, scope, "Intl.NumberFormat.prototype.formatToParts called on value that's not a NumberFormat"_s));

    auto value = toIntlMathematicalValue(globalObject, callFrame->argument(0));
    RETURN_IF_EXCEPTION(scope, { });

    if (auto number = value.tryGetDouble())
        RELEASE_AND_RETURN(scope, JSValue::encode(numberFormat->formatToParts(globalObject, number.value())));

    RELEASE_AND_RETURN(scope, JSValue::encode(numberFormat->formatToParts(globalObject, WTFMove(value))));
}

JSC_DEFINE_HOST_FUNCTION(intlNumberFormatPrototypeFuncFormatRangeToParts, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    // Do not use unwrapForOldFunctions.
    auto* numberFormat = jsDynamicCast<IntlNumberFormat*>(callFrame->thisValue());
    if (UNLIKELY(!numberFormat))
        return JSValue::encode(throwTypeError(globalObject, scope, "Intl.NumberFormat.prototype.formatRangeToParts called on value that's not a NumberFormat"_s));

    JSValue startValue = callFrame->argument(0);
    JSValue endValue = callFrame->argument(1);

    if (startValue.isUndefined() || endValue.isUndefined())
        return throwVMTypeError(globalObject, scope, "start or end is undefined"_s);

    auto start = toIntlMathematicalValue(globalObject, startValue);
    RETURN_IF_EXCEPTION(scope, { });

    auto end = toIntlMathematicalValue(globalObject, endValue);
    RETURN_IF_EXCEPTION(scope, { });

    if (auto startNumber = start.tryGetDouble()) {
        if (auto endNumber = end.tryGetDouble())
            RELEASE_AND_RETURN(scope, JSValue::encode(numberFormat->formatRangeToParts(globalObject, startNumber.value(), endNumber.value())));
    }

    RELEASE_AND_RETURN(scope, JSValue::encode(numberFormat->formatRangeToParts(globalObject, WTFMove(start), WTFMove(end))));
}

JSC_DEFINE_HOST_FUNCTION(intlNumberFormatPrototypeFuncResolvedOptions, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    // 11.3.5 Intl.NumberFormat.prototype.resolvedOptions() (ECMA-402 2.0)

    auto* numberFormat = IntlNumberFormat::unwrapForOldFunctions(globalObject, callFrame->thisValue());
    RETURN_IF_EXCEPTION(scope, { });
    if (UNLIKELY(!numberFormat))
        return JSValue::encode(throwTypeError(globalObject, scope, "Intl.NumberFormat.prototype.resolvedOptions called on value that's not a NumberFormat"_s));

    RELEASE_AND_RETURN(scope, JSValue::encode(numberFormat->resolvedOptions(globalObject)));
}

} // namespace JSC
