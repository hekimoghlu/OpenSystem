/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 22, 2023.
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
#include "JSIterator.h"

#include "AbstractSlotVisitor.h"
#include "JSCInlines.h"
#include "JSIteratorConstructor.h"
#include "SlotVisitor.h"

namespace JSC {

const ClassInfo JSIterator::s_info = { "Iterator"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSIterator) };

Structure* JSIterator::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(JSIteratorType, StructureFlags), info());
}

JSIterator* JSIterator::create(VM& vm, Structure* structure)
{
    JSIterator* instance = new (NotNull, allocateCell<JSIterator>(vm)) JSIterator(vm, structure);
    instance->finishCreation(vm);
    return instance;
}

template<typename Visitor>
void JSIterator::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    auto* thisObject = jsCast<JSIterator*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
}

DEFINE_VISIT_CHILDREN(JSIterator);

} // namespace JSC
