/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 8, 2022.
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
#include "JSWrapForValidIterator.h"


#include "JSCInlines.h"
#include "JSInternalFieldObjectImplInlines.h"

namespace JSC {

const ClassInfo JSWrapForValidIterator::s_info = { "Iterator"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSWrapForValidIterator) };

JSWrapForValidIterator* JSWrapForValidIterator::createWithInitialValues(VM& vm, Structure* structure)
{
    auto values = initialValues();
    JSWrapForValidIterator* iterator = new (NotNull, allocateCell<JSWrapForValidIterator>(vm)) JSWrapForValidIterator(vm, structure);
    iterator->finishCreation(vm, values[0], values[1]);
    return iterator;
}

JSWrapForValidIterator* JSWrapForValidIterator::create(VM& vm, Structure* structure, JSValue iterator, JSValue nextMethod)
{
    JSWrapForValidIterator* result = new (NotNull, allocateCell<JSWrapForValidIterator>(vm)) JSWrapForValidIterator(vm, structure);
    result->finishCreation(vm, iterator, nextMethod);
    return result;
}

void JSWrapForValidIterator::finishCreation(VM& vm, JSValue iterator, JSValue nextMethod)
{
    Base::finishCreation(vm);
    this->setIteratedIterator(vm, iterator);
    this->setIteratedNextMethod(vm, nextMethod);
}

template<typename Visitor>
void JSWrapForValidIterator::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    auto* thisObject = jsCast<JSWrapForValidIterator*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
}

DEFINE_VISIT_CHILDREN(JSWrapForValidIterator);

JSC_DEFINE_HOST_FUNCTION(wrapForValidIteratorPrivateFuncCreate, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return JSValue::encode(JSWrapForValidIterator::create(globalObject->vm(), globalObject->wrapForValidIteratorStructure(), callFrame->uncheckedArgument(0), callFrame->uncheckedArgument(1)));
}

} // namespace JSC
