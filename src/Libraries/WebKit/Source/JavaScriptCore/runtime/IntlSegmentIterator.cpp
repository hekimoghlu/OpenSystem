/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 23, 2021.
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
#include "IntlSegmentIterator.h"

#include "IteratorOperations.h"
#include "JSCInlines.h"
#include "ObjectConstructor.h"
#include <unicode/ucurr.h>
#include <unicode/uloc.h>
#include <wtf/unicode/icu/ICUHelpers.h>

namespace JSC {

const ClassInfo IntlSegmentIterator::s_info = { "Object"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(IntlSegmentIterator) };

IntlSegmentIterator* IntlSegmentIterator::create(VM& vm, Structure* structure, std::unique_ptr<UBreakIterator, UBreakIteratorDeleter>&& segmenter, Box<Vector<UChar>> buffer, JSString* string, IntlSegmenter::Granularity granularity)
{
    auto* object = new (NotNull, allocateCell<IntlSegmentIterator>(vm)) IntlSegmentIterator(vm, structure, WTFMove(segmenter), WTFMove(buffer), granularity, string);
    object->finishCreation(vm);
    return object;
}

Structure* IntlSegmentIterator::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
}

IntlSegmentIterator::IntlSegmentIterator(VM& vm, Structure* structure, std::unique_ptr<UBreakIterator, UBreakIteratorDeleter>&& segmenter, Box<Vector<UChar>>&& buffer, IntlSegmenter::Granularity granularity, JSString* string)
    : Base(vm, structure)
    , m_segmenter(WTFMove(segmenter))
    , m_buffer(WTFMove(buffer))
    , m_string(string, WriteBarrierEarlyInit)
    , m_granularity(granularity)
{
}

template<typename Visitor>
void IntlSegmentIterator::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    auto* thisObject = jsCast<IntlSegmentIterator*>(cell);
    Base::visitChildren(thisObject, visitor);
    visitor.append(thisObject->m_string);
}

DEFINE_VISIT_CHILDREN(IntlSegmentIterator);

JSObject* IntlSegmentIterator::next(JSGlobalObject* globalObject)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    int32_t startIndex = ubrk_current(m_segmenter.get());
    int32_t endIndex = ubrk_next(m_segmenter.get());
    if (endIndex == UBRK_DONE)
        return createIteratorResultObject(globalObject, jsUndefined(), true);
    JSObject* object = IntlSegmenter::createSegmentDataObject(globalObject, m_string.get(), startIndex, endIndex, *m_segmenter, m_granularity);
    RETURN_IF_EXCEPTION(scope, { });
    return createIteratorResultObject(globalObject, object, false);
}

} // namespace JSC
