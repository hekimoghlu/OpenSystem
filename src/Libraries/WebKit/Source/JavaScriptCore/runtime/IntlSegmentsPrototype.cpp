/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 30, 2023.
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
#include "IntlSegmentsPrototype.h"

#include "IntlSegments.h"
#include "JSCInlines.h"

namespace JSC {

static JSC_DECLARE_HOST_FUNCTION(intlSegmentsPrototypeFuncContaining);
static JSC_DECLARE_HOST_FUNCTION(intlSegmentsPrototypeFuncIterator);

}

#include "IntlSegmentsPrototype.lut.h"

namespace JSC {

const ClassInfo IntlSegmentsPrototype::s_info = { "%Segments%"_s, &Base::s_info, &segmentsPrototypeTable, nullptr, CREATE_METHOD_TABLE(IntlSegmentsPrototype) };

/* Source for IntlSegmentsPrototype.lut.h
@begin segmentsPrototypeTable
  containing       intlSegmentsPrototypeFuncContaining         DontEnum|Function 1
@end
*/

IntlSegmentsPrototype* IntlSegmentsPrototype::create(VM& vm, JSGlobalObject* globalObject, Structure* structure)
{
    auto* object = new (NotNull, allocateCell<IntlSegmentsPrototype>(vm)) IntlSegmentsPrototype(vm, structure);
    object->finishCreation(vm, globalObject);
    return object;
}

Structure* IntlSegmentsPrototype::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
}

IntlSegmentsPrototype::IntlSegmentsPrototype(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

void IntlSegmentsPrototype::finishCreation(VM& vm, JSGlobalObject* globalObject)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->iteratorSymbol, intlSegmentsPrototypeFuncIterator, static_cast<unsigned>(PropertyAttribute::DontEnum), 0, ImplementationVisibility::Public);
}

// https://tc39.es/proposal-intl-segmenter/#sec-%segmentsprototype%.containing
JSC_DEFINE_HOST_FUNCTION(intlSegmentsPrototypeFuncContaining, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* segments = jsDynamicCast<IntlSegments*>(callFrame->thisValue());
    if (UNLIKELY(!segments))
        return throwVMTypeError(globalObject, scope, "%Segments.prototype%.containing called on value that's not a Segments"_s);

    RELEASE_AND_RETURN(scope, JSValue::encode(segments->containing(globalObject, callFrame->argument(0))));
}

// https://tc39.es/proposal-intl-segmenter/#sec-%segmentsprototype%-@@iterator
JSC_DEFINE_HOST_FUNCTION(intlSegmentsPrototypeFuncIterator, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* segments = jsDynamicCast<IntlSegments*>(callFrame->thisValue());
    if (UNLIKELY(!segments))
        return throwVMTypeError(globalObject, scope, "%Segments.prototype%[@@iterator] called on value that's not a Segments"_s);

    RELEASE_AND_RETURN(scope, JSValue::encode(segments->createSegmentIterator(globalObject)));
}

} // namespace JSC
