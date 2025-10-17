/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 6, 2025.
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
#include "JSAsyncFromSyncIterator.h"

#include "JSCInlines.h"
#include "JSInternalFieldObjectImplInlines.h"

namespace JSC {

const ClassInfo JSAsyncFromSyncIterator::s_info = { "AsyncFromSyncIterator"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSAsyncFromSyncIterator) };

void JSAsyncFromSyncIterator::finishCreation(VM& vm, JSValue syncIterator, JSValue nextMethod)
{
    Base::finishCreation(vm);
    this->setSyncIterator(vm, syncIterator);
    this->setNextMethod(vm, nextMethod);
}

JSAsyncFromSyncIterator* JSAsyncFromSyncIterator::createWithInitialValues(VM& vm, Structure* structure)
{
    auto values = initialValues();
    JSAsyncFromSyncIterator* result = new (NotNull, allocateCell<JSAsyncFromSyncIterator>(vm)) JSAsyncFromSyncIterator(vm, structure);
    result->finishCreation(vm, values[0], values[1]);
    return result;
}

JSAsyncFromSyncIterator* JSAsyncFromSyncIterator::create(VM& vm, Structure* structure, JSValue syncIterator, JSValue nextMethod)
{
    ASSERT(syncIterator.isObject());
    JSAsyncFromSyncIterator* result = new (NotNull, allocateCell<JSAsyncFromSyncIterator>(vm)) JSAsyncFromSyncIterator(vm, structure);
    result->finishCreation(vm, syncIterator, nextMethod);
    return result;
}

template<typename Visitor>
void JSAsyncFromSyncIterator::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    auto* thisObject = jsCast<JSAsyncFromSyncIterator*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
}

DEFINE_VISIT_CHILDREN(JSAsyncFromSyncIterator);

JSC_DEFINE_HOST_FUNCTION(asyncFromSyncIteratorPrivateFuncCreate, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    ASSERT(callFrame->uncheckedArgument(0).isObject());
    return JSValue::encode(JSAsyncFromSyncIterator::create(globalObject->vm(), globalObject->asyncFromSyncIteratorStructure(), callFrame->uncheckedArgument(0), callFrame->uncheckedArgument(1)));
}

} // namespace JSC
