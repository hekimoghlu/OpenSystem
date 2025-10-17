/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 9, 2024.
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
#include "JSRegExpStringIterator.h"

#include "Error.h"
#include "JSCInlines.h"
#include "JSCJSValue.h"
#include "JSInternalFieldObjectImplInlines.h"

namespace JSC {

const ClassInfo JSRegExpStringIterator::s_info = { "RegExpStringIterator"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSRegExpStringIterator) };

JSRegExpStringIterator* JSRegExpStringIterator::createWithInitialValues(VM& vm, Structure* structure)
{
    JSRegExpStringIterator* iterator = new (NotNull, allocateCell<JSRegExpStringIterator>(vm)) JSRegExpStringIterator(vm, structure);
    iterator->finishCreation(vm);
    return iterator;
}

void JSRegExpStringIterator::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    auto values = initialValues();
    for (unsigned index = 0; index < values.size(); ++index)
        Base::internalField(index).set(vm, this, values[index]);
}

template<typename Visitor>
void JSRegExpStringIterator::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    auto* thisObject = jsCast<JSRegExpStringIterator*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
}

DEFINE_VISIT_CHILDREN(JSRegExpStringIterator);

JSC_DEFINE_HOST_FUNCTION(regExpStringIteratorPrivateFuncCreate, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    ASSERT(callFrame->argument(0).isCell());
    ASSERT(callFrame->argument(1).isString());
    ASSERT(callFrame->argument(2).isBoolean());
    ASSERT(callFrame->argument(3).isBoolean());

    VM& vm = globalObject->vm();

    auto* regExpStringIterator = JSRegExpStringIterator::createWithInitialValues(vm, globalObject->regExpStringIteratorStructure());

    regExpStringIterator->setRegExp(vm, asObject(callFrame->uncheckedArgument(0)));
    regExpStringIterator->setString(vm, callFrame->uncheckedArgument(1));
    regExpStringIterator->setGlobal(vm, callFrame->argument(2));
    regExpStringIterator->setFullUnicode(vm, callFrame->argument(3));

    return JSValue::encode(regExpStringIterator);
}

} // namespace JSC
