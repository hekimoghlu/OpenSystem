/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 1, 2023.
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
#include "JSIteratorHelper.h"

#include "JSCInlines.h"
#include "JSInternalFieldObjectImplInlines.h"

namespace JSC {

const ClassInfo JSIteratorHelper::s_info = { "Iterator Helper"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSIteratorHelper) };

void JSIteratorHelper::finishCreation(VM& vm, JSValue generator, JSValue underlyingIterator)
{
    Base::finishCreation(vm);
    internalField(Field::Generator).set(vm, this, generator);
    internalField(Field::UnderlyingIterator).set(vm, this, underlyingIterator);
}

JSIteratorHelper* JSIteratorHelper::createWithInitialValues(VM& vm, Structure* structure)
{
    auto values = initialValues();
    JSIteratorHelper* result = new (NotNull, allocateCell<JSIteratorHelper>(vm)) JSIteratorHelper(vm, structure);
    result->finishCreation(vm, values[0], values[1]);
    return result;
}

JSIteratorHelper* JSIteratorHelper::create(VM& vm, Structure* structure, JSValue generator, JSValue underlyingIterator)
{
    ASSERT(generator.isObject() && (underlyingIterator.isObject() || underlyingIterator.isNull()));
    JSIteratorHelper* result = new (NotNull, allocateCell<JSIteratorHelper>(vm)) JSIteratorHelper(vm, structure);
    result->finishCreation(vm, generator, underlyingIterator);
    return result;
}

Structure* JSIteratorHelper::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(JSIteratorHelperType, StructureFlags), info());
}

JSIteratorHelper::JSIteratorHelper(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

template<typename Visitor>
void JSIteratorHelper::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    auto* thisObject = jsCast<JSIteratorHelper*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
}

DEFINE_VISIT_CHILDREN(JSIteratorHelper);

JSC_DEFINE_HOST_FUNCTION(iteratorHelperPrivateFuncCreate, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return JSValue::encode(JSIteratorHelper::create(globalObject->vm(), globalObject->iteratorHelperStructure(), callFrame->uncheckedArgument(0), callFrame->uncheckedArgument(1)));
}

} // namespace JSC
