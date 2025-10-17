/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 8, 2023.
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
#include "GeneratorFunctionConstructor.h"

#include "FunctionConstructor.h"
#include "GeneratorFunctionPrototype.h"
#include "JSCInlines.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(GeneratorFunctionConstructor);

const ClassInfo GeneratorFunctionConstructor::s_info = { "GeneratorFunction"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(GeneratorFunctionConstructor) };

static JSC_DECLARE_HOST_FUNCTION(callGeneratorFunctionConstructor);
static JSC_DECLARE_HOST_FUNCTION(constructGeneratorFunctionConstructor);

JSC_DEFINE_HOST_FUNCTION(callGeneratorFunctionConstructor, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    ArgList args(callFrame);
    return JSValue::encode(constructFunction(globalObject, callFrame, args, FunctionConstructionMode::Generator));
}

JSC_DEFINE_HOST_FUNCTION(constructGeneratorFunctionConstructor, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    ArgList args(callFrame);
    return JSValue::encode(constructFunction(globalObject, callFrame, args, FunctionConstructionMode::Generator, callFrame->newTarget()));
}

GeneratorFunctionConstructor::GeneratorFunctionConstructor(VM& vm, Structure* structure)
    : InternalFunction(vm, structure, callGeneratorFunctionConstructor, constructGeneratorFunctionConstructor)
{
}

void GeneratorFunctionConstructor::finishCreation(VM& vm, GeneratorFunctionPrototype* generatorFunctionPrototype)
{
    Base::finishCreation(vm, 1, "GeneratorFunction"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, generatorFunctionPrototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
}

} // namespace JSC
