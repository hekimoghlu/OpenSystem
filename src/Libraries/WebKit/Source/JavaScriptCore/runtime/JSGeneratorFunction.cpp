/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 5, 2021.
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
#include "JSGeneratorFunction.h"

#include "JSCellInlines.h"
#include "JSFunctionInlines.h"
#include "VM.h"

namespace JSC {

const ClassInfo JSGeneratorFunction::s_info = { "GeneratorFunction"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSGeneratorFunction) };

JSGeneratorFunction::JSGeneratorFunction(VM& vm, FunctionExecutable* executable, JSScope* scope, Structure* structure)
    : Base(vm, executable, scope, structure)
{
}

JSGeneratorFunction* JSGeneratorFunction::createImpl(VM& vm, FunctionExecutable* executable, JSScope* scope, Structure* structure)
{
    JSGeneratorFunction* generatorFunction = new (NotNull, allocateCell<JSGeneratorFunction>(vm)) JSGeneratorFunction(vm, executable, scope, structure);
    ASSERT(generatorFunction->structure()->globalObject());
    generatorFunction->finishCreation(vm);
    return generatorFunction;
}

JSGeneratorFunction* JSGeneratorFunction::create(VM& vm, JSGlobalObject* globalObject, FunctionExecutable* executable, JSScope* scope)
{
    return create(vm, globalObject, executable, scope, globalObject->generatorFunctionStructure());
}

JSGeneratorFunction* JSGeneratorFunction::create(VM& vm, JSGlobalObject*, FunctionExecutable* executable, JSScope* scope, Structure* structure)
{
    JSGeneratorFunction* generatorFunction = createImpl(vm, executable, scope, structure);
    executable->notifyCreation(vm, generatorFunction, "Allocating a generator function");
    return generatorFunction;
}

JSGeneratorFunction* JSGeneratorFunction::createWithInvalidatedReallocationWatchpoint(VM& vm, JSGlobalObject* globalObject, FunctionExecutable* executable, JSScope* scope)
{
    return createWithInvalidatedReallocationWatchpoint(vm, globalObject, executable, scope, globalObject->generatorFunctionStructure());
}

JSGeneratorFunction* JSGeneratorFunction::createWithInvalidatedReallocationWatchpoint(VM& vm, JSGlobalObject*, FunctionExecutable* executable, JSScope* scope, Structure* structure)
{
    return createImpl(vm, executable, scope, structure);
}

} // namespace JSC
