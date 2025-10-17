/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 1, 2024.
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
#include "IntlDisplayNamesPrototype.h"

#include "IntlDisplayNames.h"
#include "JSCInlines.h"

namespace JSC {

static JSC_DECLARE_HOST_FUNCTION(intlDisplayNamesPrototypeFuncOf);
static JSC_DECLARE_HOST_FUNCTION(intlDisplayNamesPrototypeFuncResolvedOptions);

}

#include "IntlDisplayNamesPrototype.lut.h"

namespace JSC {

const ClassInfo IntlDisplayNamesPrototype::s_info = { "Intl.DisplayNames"_s, &Base::s_info, &displayNamesPrototypeTable, nullptr, CREATE_METHOD_TABLE(IntlDisplayNamesPrototype) };

/* Source for IntlDisplayNamesPrototype.lut.h
@begin displayNamesPrototypeTable
  of               intlDisplayNamesPrototypeFuncOf                 DontEnum|Function 1
  resolvedOptions  intlDisplayNamesPrototypeFuncResolvedOptions    DontEnum|Function 0
@end
*/

IntlDisplayNamesPrototype* IntlDisplayNamesPrototype::create(VM& vm, Structure* structure)
{
    auto* object = new (NotNull, allocateCell<IntlDisplayNamesPrototype>(vm)) IntlDisplayNamesPrototype(vm, structure);
    object->finishCreation(vm);
    return object;
}

Structure* IntlDisplayNamesPrototype::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
}

IntlDisplayNamesPrototype::IntlDisplayNamesPrototype(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

void IntlDisplayNamesPrototype::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

// https://tc39.es/proposal-intl-displaynames/#sec-Intl.DisplayNames.prototype.of
JSC_DEFINE_HOST_FUNCTION(intlDisplayNamesPrototypeFuncOf, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* displayNames = jsDynamicCast<IntlDisplayNames*>(callFrame->thisValue());
    if (UNLIKELY(!displayNames))
        return throwVMTypeError(globalObject, scope, "Intl.DisplayNames.prototype.of called on value that's not a DisplayNames"_s);

    RELEASE_AND_RETURN(scope, JSValue::encode(displayNames->of(globalObject, callFrame->argument(0))));
}

// https://tc39.es/proposal-intl-displaynames/#sec-Intl.DisplayNames.prototype.resolvedOptions
JSC_DEFINE_HOST_FUNCTION(intlDisplayNamesPrototypeFuncResolvedOptions, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* displayNames = jsDynamicCast<IntlDisplayNames*>(callFrame->thisValue());
    if (UNLIKELY(!displayNames))
        return throwVMTypeError(globalObject, scope, "Intl.DisplayNames.prototype.resolvedOptions called on value that's not a DisplayNames"_s);

    RELEASE_AND_RETURN(scope, JSValue::encode(displayNames->resolvedOptions(globalObject)));
}

} // namespace JSC
