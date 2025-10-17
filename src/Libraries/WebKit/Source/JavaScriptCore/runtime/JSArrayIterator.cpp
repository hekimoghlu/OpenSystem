/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 4, 2023.
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
#include "JSArrayIterator.h"

#include "JSCInlines.h"
#include "JSInternalFieldObjectImplInlines.h"

namespace JSC {

const ClassInfo JSArrayIterator::s_info = { "ArrayIterator"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSArrayIterator) };

JSArrayIterator* JSArrayIterator::create(VM& vm, Structure* structure, JSObject* iteratedObject, JSValue kind)
{
    ASSERT(kind.asUInt32() <= static_cast<uint32_t>(IterationKind::Entries));
    JSArrayIterator* iterator = new (NotNull, allocateCell<JSArrayIterator>(vm)) JSArrayIterator(vm, structure);
    iterator->finishCreation(vm);
    iterator->internalField(Field::IteratedObject).set(vm, iterator, iteratedObject);
    iterator->internalField(Field::Kind).set(vm, iterator, kind);
    return iterator;
}

JSArrayIterator* JSArrayIterator::createWithInitialValues(VM& vm, Structure* structure)
{
    JSArrayIterator* iterator = new (NotNull, allocateCell<JSArrayIterator>(vm)) JSArrayIterator(vm, structure);
    iterator->finishCreation(vm);
    return iterator;
}

Structure* JSArrayIterator::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(JSArrayIteratorType, StructureFlags), info());
}

JSArrayIterator::JSArrayIterator(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

void JSArrayIterator::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    auto values = initialValues();
    for (unsigned index = 0; index < values.size(); ++index)
        Base::internalField(index).set(vm, this, values[index]);
}

template<typename Visitor>
void JSArrayIterator::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    auto* thisObject = jsCast<JSArrayIterator*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
}

DEFINE_VISIT_CHILDREN(JSArrayIterator);

} // namespace JSC
