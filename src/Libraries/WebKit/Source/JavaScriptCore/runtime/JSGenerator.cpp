/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 23, 2023.
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
#include "JSGenerator.h"

#include "JSCInlines.h"
#include "JSInternalFieldObjectImplInlines.h"

namespace JSC {

const ClassInfo JSGenerator::s_info = { "Generator"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSGenerator) };

JSGenerator* JSGenerator::create(VM& vm, Structure* structure)
{
    JSGenerator* generator = new (NotNull, allocateCell<JSGenerator>(vm)) JSGenerator(vm, structure);
    generator->finishCreation(vm);
    return generator;
}

Structure* JSGenerator::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(JSGeneratorType, StructureFlags), info());
}

JSGenerator::JSGenerator(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

void JSGenerator::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    auto values = initialValues();
    ASSERT(values.size() == numberOfInternalFields);
    for (unsigned index = 0; index < values.size(); ++index)
        internalField(index).set(vm, this, values[index]);
}

template<typename Visitor>
void JSGenerator::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    auto* thisObject = jsCast<JSGenerator*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
}

DEFINE_VISIT_CHILDREN(JSGenerator);

} // namespace JSC
