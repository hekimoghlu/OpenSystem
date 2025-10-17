/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 22, 2022.
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
#include "IntlSegmentIteratorPrototype.h"

#include "IntlSegmentIterator.h"
#include "JSCInlines.h"

namespace JSC {

static JSC_DECLARE_HOST_FUNCTION(intlSegmentIteratorPrototypeFuncNext);

}

#include "IntlSegmentIteratorPrototype.lut.h"

namespace JSC {

const ClassInfo IntlSegmentIteratorPrototype::s_info = { "Segment String Iterator"_s, &Base::s_info, &segmentIteratorPrototypeTable, nullptr, CREATE_METHOD_TABLE(IntlSegmentIteratorPrototype) };

/* Source for IntlSegmentIteratorPrototype.lut.h
@begin segmentIteratorPrototypeTable
  next             intlSegmentIteratorPrototypeFuncNext               DontEnum|Function 0
@end
*/

IntlSegmentIteratorPrototype* IntlSegmentIteratorPrototype::create(VM& vm, Structure* structure)
{
    auto* object = new (NotNull, allocateCell<IntlSegmentIteratorPrototype>(vm)) IntlSegmentIteratorPrototype(vm, structure);
    object->finishCreation(vm);
    return object;
}

Structure* IntlSegmentIteratorPrototype::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
}

IntlSegmentIteratorPrototype::IntlSegmentIteratorPrototype(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

void IntlSegmentIteratorPrototype::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

// https://tc39.es/proposal-intl-segmenter/#sec-%segmentiteratorprototype%.next
JSC_DEFINE_HOST_FUNCTION(intlSegmentIteratorPrototypeFuncNext, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* segmentIterator = jsDynamicCast<IntlSegmentIterator*>(callFrame->thisValue());
    if (UNLIKELY(!segmentIterator))
        return throwVMTypeError(globalObject, scope, "Intl.SegmentIterator.prototype.next called on value that's not a SegmentIterator"_s);

    RELEASE_AND_RETURN(scope, JSValue::encode(segmentIterator->next(globalObject)));
}

} // namespace JSC
