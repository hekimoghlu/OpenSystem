/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 26, 2024.
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
#include "JSMapIterator.h"

#include "JSCInlines.h"
#include "JSInternalFieldObjectImplInlines.h"
#include "JSMap.h"

namespace JSC {

const ClassInfo JSMapIterator::s_info = { "Map Iterator"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSMapIterator) };

JSMapIterator* JSMapIterator::createWithInitialValues(VM& vm, Structure* structure)
{
    JSMapIterator* iterator = new (NotNull, allocateCell<JSMapIterator>(vm)) JSMapIterator(vm, structure);
    iterator->finishCreation(vm);
    return iterator;
}

void JSMapIterator::finishCreation(JSGlobalObject* globalObject, JSMap* iteratedObject, IterationKind kind)
{
    VM& vm = getVM(globalObject);
    auto scope = DECLARE_THROW_SCOPE(vm);

    Base::finishCreation(vm);
    setEntry(vm, 0);
    setIteratedObject(vm, iteratedObject);

    JSCell* storage = iteratedObject->storage(globalObject);
    RETURN_IF_EXCEPTION(scope, void());
    setStorage(vm, storage);

    internalField(Field::Kind).set(vm, this, jsNumber(static_cast<int32_t>(kind)));
}

void JSMapIterator::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    auto values = initialValues();
    for (unsigned index = 0; index < values.size(); ++index)
        Base::internalField(index).set(vm, this, values[index]);
}

template<typename Visitor>
void JSMapIterator::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    auto* thisObject = jsCast<JSMapIterator*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
}

DEFINE_VISIT_CHILDREN(JSMapIterator);


JSC_DEFINE_HOST_FUNCTION(mapIteratorPrivateFuncMapIteratorNext, (JSGlobalObject * globalObject, CallFrame* callFrame))
{
    ASSERT(callFrame->argument(0).isCell());

    VM& vm = globalObject->vm();
    JSCell* cell = callFrame->uncheckedArgument(0).asCell();
    if (cell == vm.orderedHashTableSentinel())
        return JSValue::encode(cell);
    return JSValue::encode(jsCast<JSMapIterator*>(cell)->next(vm));
}

JSC_DEFINE_HOST_FUNCTION(mapIteratorPrivateFuncMapIteratorKey, (JSGlobalObject * globalObject, CallFrame* callFrame))
{
    ASSERT(callFrame->argument(0).isCell());

    VM& vm = globalObject->vm();
    JSCell* cell = callFrame->uncheckedArgument(0).asCell();
    if (cell == vm.orderedHashTableSentinel())
        return JSValue::encode(cell);
    return JSValue::encode(jsCast<JSMapIterator*>(cell)->nextKey(vm));
}

JSC_DEFINE_HOST_FUNCTION(mapIteratorPrivateFuncMapIteratorValue, (JSGlobalObject * globalObject, CallFrame* callFrame))
{
    ASSERT(callFrame->argument(0).isCell());

    VM& vm = globalObject->vm();
    JSCell* cell = callFrame->uncheckedArgument(0).asCell();
    if (cell == vm.orderedHashTableSentinel())
        return JSValue::encode(cell);
    return JSValue::encode(jsCast<JSMapIterator*>(cell)->nextValue(vm));
}

}
