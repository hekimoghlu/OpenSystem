/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 4, 2024.
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
#include "IntlListFormatPrototype.h"

#include "IntlListFormat.h"
#include "JSCInlines.h"

namespace JSC {

static JSC_DECLARE_HOST_FUNCTION(intlListFormatPrototypeFuncFormat);
static JSC_DECLARE_HOST_FUNCTION(intlListFormatPrototypeFuncFormatToParts);
static JSC_DECLARE_HOST_FUNCTION(intlListFormatPrototypeFuncResolvedOptions);

}

#include "IntlListFormatPrototype.lut.h"

namespace JSC {

const ClassInfo IntlListFormatPrototype::s_info = { "Intl.ListFormat"_s, &Base::s_info, &listFormatPrototypeTable, nullptr, CREATE_METHOD_TABLE(IntlListFormatPrototype) };

/* Source for IntlListFormatPrototype.lut.h
@begin listFormatPrototypeTable
  format           intlListFormatPrototypeFuncFormat             DontEnum|Function 1
  formatToParts    intlListFormatPrototypeFuncFormatToParts      DontEnum|Function 1
  resolvedOptions  intlListFormatPrototypeFuncResolvedOptions    DontEnum|Function 0
@end
*/

IntlListFormatPrototype* IntlListFormatPrototype::create(VM& vm, Structure* structure)
{
    auto* object = new (NotNull, allocateCell<IntlListFormatPrototype>(vm)) IntlListFormatPrototype(vm, structure);
    object->finishCreation(vm);
    return object;
}

Structure* IntlListFormatPrototype::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
}

IntlListFormatPrototype::IntlListFormatPrototype(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

void IntlListFormatPrototype::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

// https://tc39.es/proposal-intl-list-format/#sec-Intl.ListFormat.prototype.format
JSC_DEFINE_HOST_FUNCTION(intlListFormatPrototypeFuncFormat, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* listFormat = jsDynamicCast<IntlListFormat*>(callFrame->thisValue());
    if (UNLIKELY(!listFormat))
        return throwVMTypeError(globalObject, scope, "Intl.ListFormat.prototype.format called on value that's not a ListFormat"_s);

    RELEASE_AND_RETURN(scope, JSValue::encode(listFormat->format(globalObject, callFrame->argument(0))));
}

// https://tc39.es/proposal-intl-list-format/#sec-Intl.ListFormat.prototype.formatToParts
JSC_DEFINE_HOST_FUNCTION(intlListFormatPrototypeFuncFormatToParts, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* listFormat = jsDynamicCast<IntlListFormat*>(callFrame->thisValue());
    if (UNLIKELY(!listFormat))
        return throwVMTypeError(globalObject, scope, "Intl.ListFormat.prototype.formatToParts called on value that's not a ListFormat"_s);

    RELEASE_AND_RETURN(scope, JSValue::encode(listFormat->formatToParts(globalObject, callFrame->argument(0))));
}

// https://tc39.es/proposal-intl-list-format/#sec-Intl.ListFormat.prototype.resolvedOptions
JSC_DEFINE_HOST_FUNCTION(intlListFormatPrototypeFuncResolvedOptions, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* listFormat = jsDynamicCast<IntlListFormat*>(callFrame->thisValue());
    if (UNLIKELY(!listFormat))
        return throwVMTypeError(globalObject, scope, "Intl.ListFormat.prototype.resolvedOptions called on value that's not a ListFormat"_s);

    RELEASE_AND_RETURN(scope, JSValue::encode(listFormat->resolvedOptions(globalObject)));
}

} // namespace JSC
