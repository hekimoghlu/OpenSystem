/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 8, 2025.
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
#include "JSSetIterator.h"

#include "JSCInlines.h"
#include "JSInternalFieldObjectImplInlines.h"
#include "JSSet.h"

namespace JSC {

const ClassInfo JSSetIterator::s_info = { "Set Iterator"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSSetIterator) };

JSSetIterator* JSSetIterator::createWithInitialValues(VM& vm, Structure* structure)
{
    JSSetIterator* iterator = new (NotNull, allocateCell<JSSetIterator>(vm)) JSSetIterator(vm, structure);
    iterator->finishCreation(vm);
    return iterator;
}

void JSSetIterator::finishCreation(JSGlobalObject* globalObject,  JSSet* iteratedObject, IterationKind kind)
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

void JSSetIterator::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    auto values = initialValues();
    for (unsigned index = 0; index < values.size(); ++index)
        Base::internalField(index).set(vm, this, values[index]);
}

template<typename Visitor>
void JSSetIterator::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    auto* thisObject = jsCast<JSSetIterator*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
}

DEFINE_VISIT_CHILDREN(JSSetIterator);

JSC_DEFINE_HOST_FUNCTION(setIteratorPrivateFuncSetIteratorNext, (JSGlobalObject * globalObject, CallFrame* callFrame))
{
    ASSERT(callFrame->argument(0).isCell());

    VM& vm = globalObject->vm();
    JSCell* cell = callFrame->uncheckedArgument(0).asCell();
    if (cell == vm.orderedHashTableSentinel())
        return JSValue::encode(cell);

    JSSetIterator* iterator = jsCast<JSSetIterator*>(cell);
    return JSValue::encode(iterator->next(vm));
}

JSC_DEFINE_HOST_FUNCTION(setIteratorPrivateFuncSetIteratorKey, (JSGlobalObject * globalObject, CallFrame* callFrame))
{
    ASSERT(callFrame->argument(0).isCell());

    VM& vm = globalObject->vm();
    JSCell* cell = callFrame->uncheckedArgument(0).asCell();
    if (cell == vm.orderedHashTableSentinel())
        return JSValue::encode(cell);

    JSSetIterator* iterator = jsCast<JSSetIterator*>(cell);
    return JSValue::encode(iterator->nextKey(vm));
}

}
