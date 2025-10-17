/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 6, 2023.
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
#include "JSStringIterator.h"

#include "JSCInlines.h"
#include "JSInternalFieldObjectImplInlines.h"

namespace JSC {

const ClassInfo JSStringIterator::s_info = { "String Iterator"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSStringIterator) };

void JSStringIterator::finishCreation(VM& vm, JSString* iteratedString)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    internalField(Field::Index).set(vm, this, jsNumber(0));
    internalField(Field::IteratedString).set(vm, this, iteratedString);
}

JSStringIterator* JSStringIterator::clone(JSGlobalObject* globalObject)
{
    VM& vm = globalObject->vm();
    JSString* iteratedString = jsCast<JSString*>(this->iteratedString());
    auto* clone = JSStringIterator::create(vm, globalObject->stringIteratorStructure(), iteratedString);
    clone->internalField(Field::Index).set(vm, clone, this->index());
    return clone;
}

template<typename Visitor>
void JSStringIterator::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    auto* thisObject = jsCast<JSStringIterator*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
}

DEFINE_VISIT_CHILDREN(JSStringIterator);

} // namespace JSC
