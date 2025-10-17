/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 10, 2024.
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
#include "IntlCollatorPrototype.h"

#include "IntlCollator.h"
#include "JSBoundFunction.h"
#include "JSCInlines.h"

namespace JSC {

static JSC_DECLARE_CUSTOM_GETTER(intlCollatorPrototypeGetterCompare);
static JSC_DECLARE_HOST_FUNCTION(intlCollatorPrototypeFuncResolvedOptions);
static JSC_DECLARE_HOST_FUNCTION(intlCollatorFuncCompare);

}

#include "IntlCollatorPrototype.lut.h"

namespace JSC {

const ClassInfo IntlCollatorPrototype::s_info = { "Intl.Collator"_s, &Base::s_info, &collatorPrototypeTable, nullptr, CREATE_METHOD_TABLE(IntlCollatorPrototype) };

/* Source for IntlCollatorPrototype.lut.h
@begin collatorPrototypeTable
  compare          intlCollatorPrototypeGetterCompare        DontEnum|ReadOnly|CustomAccessor
  resolvedOptions  intlCollatorPrototypeFuncResolvedOptions  DontEnum|Function 0
@end
*/

IntlCollatorPrototype* IntlCollatorPrototype::create(VM& vm, JSGlobalObject*, Structure* structure)
{
    IntlCollatorPrototype* object = new (NotNull, allocateCell<IntlCollatorPrototype>(vm)) IntlCollatorPrototype(vm, structure);
    object->finishCreation(vm);
    return object;
}

Structure* IntlCollatorPrototype::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
}

IntlCollatorPrototype::IntlCollatorPrototype(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

void IntlCollatorPrototype::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

JSC_DEFINE_HOST_FUNCTION(intlCollatorFuncCompare, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    // 10.3.4 Collator Compare Functions (ECMA-402 2.0)
    // 1. Let collator be the this value.
    // 2. Assert: Type(collator) is Object and collator has an [[initializedCollator]] internal slot whose value is true.
    IntlCollator* collator = jsDynamicCast<IntlCollator*>(callFrame->thisValue());
    if (UNLIKELY(!collator))
        return JSValue::encode(throwTypeError(globalObject, scope, "Intl.Collator.prototype.compare called on value that's not a Collator"_s));

    // 3. If x is not provided, let x be undefined.
    // 4. If y is not provided, let y be undefined.
    // 5. Let X be ToString(x).
    JSString* x = callFrame->argument(0).toString(globalObject);
    // 6. ReturnIfAbrupt(X).
    RETURN_IF_EXCEPTION(scope, encodedJSValue());

    // 7. Let Y be ToString(y).
    JSString* y = callFrame->argument(1).toString(globalObject);
    // 8. ReturnIfAbrupt(Y).
    RETURN_IF_EXCEPTION(scope, encodedJSValue());

    // 9. Return CompareStrings(collator, X, Y).
    auto xView = x->view(globalObject);
    RETURN_IF_EXCEPTION(scope, encodedJSValue());
    auto yView = y->view(globalObject);
    RETURN_IF_EXCEPTION(scope, encodedJSValue());
    RELEASE_AND_RETURN(scope, JSValue::encode(jsNumber(collator->compareStrings(globalObject, xView, yView))));
}

JSC_DEFINE_CUSTOM_GETTER(intlCollatorPrototypeGetterCompare, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    // 10.3.3 Intl.Collator.prototype.compare (ECMA-402 2.0)
    // 1. Let collator be this Collator object.
    IntlCollator* collator = jsDynamicCast<IntlCollator*>(JSValue::decode(thisValue));
    if (UNLIKELY(!collator))
        return JSValue::encode(throwTypeError(globalObject, scope, "Intl.Collator.prototype.compare called on value that's not a Collator"_s));

    JSBoundFunction* boundCompare = collator->boundCompare();
    // 2. If collator.[[boundCompare]] is undefined,
    if (!boundCompare) {
        JSGlobalObject* globalObject = collator->globalObject();
        // a. Let F be a new built-in function object as defined in 11.3.4.
        // b. The value of Fâ€™s length property is 2.
        JSFunction* targetObject = JSFunction::create(vm, globalObject, 2, "compare"_s, intlCollatorFuncCompare, ImplementationVisibility::Public);

        // c. Let bc be BoundFunctionCreate(F, Â«this valueÂ»).
        boundCompare = JSBoundFunction::create(vm, globalObject, targetObject, collator, { }, 2, jsEmptyString(vm));
        RETURN_IF_EXCEPTION(scope, { });
        boundCompare->reifyLazyPropertyIfNeeded<>(vm, globalObject, vm.propertyNames->name);
        RETURN_IF_EXCEPTION(scope, { });
        boundCompare->putDirect(vm, vm.propertyNames->name, jsEmptyString(vm), PropertyAttribute::ReadOnly | PropertyAttribute::DontEnum);
        // d. Set collator.[[boundCompare]] to bc.
        collator->setBoundCompare(vm, boundCompare);
    }
    // 3. Return collator.[[boundCompare]].
    return JSValue::encode(boundCompare);
}

JSC_DEFINE_HOST_FUNCTION(intlCollatorPrototypeFuncResolvedOptions, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    // 10.3.5 Intl.Collator.prototype.resolvedOptions() (ECMA-402 2.0)
    IntlCollator* collator = jsDynamicCast<IntlCollator*>(callFrame->thisValue());
    if (UNLIKELY(!collator))
        return JSValue::encode(throwTypeError(globalObject, scope, "Intl.Collator.prototype.resolvedOptions called on value that's not a Collator"_s));

    RELEASE_AND_RETURN(scope, JSValue::encode(collator->resolvedOptions(globalObject)));
}

} // namespace JSC
