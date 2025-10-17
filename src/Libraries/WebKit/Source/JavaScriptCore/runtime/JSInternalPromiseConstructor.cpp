/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 18, 2023.
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
#include "JSInternalPromiseConstructor.h"

#include "JSCBuiltins.h"
#include "JSInternalPromisePrototype.h"
#include "JSObjectInlines.h"
#include "StructureInlines.h"

#include "JSInternalPromiseConstructor.lut.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(JSInternalPromiseConstructor);

const ClassInfo JSInternalPromiseConstructor::s_info = { "Function"_s, &Base::s_info, &internalPromiseConstructorTable, nullptr, CREATE_METHOD_TABLE(JSInternalPromiseConstructor) };

/* Source for JSInternalPromiseConstructor.lut.h
@begin internalPromiseConstructorTable
  internalAll  JSBuiltin DontEnum|Function 1
@end
*/

JSInternalPromiseConstructor* JSInternalPromiseConstructor::create(VM& vm, Structure* structure, JSInternalPromisePrototype* promisePrototype)
{
    JSGlobalObject* globalObject = structure->globalObject();
    FunctionExecutable* executable = promiseConstructorInternalPromiseConstructorCodeGenerator(vm);
    JSInternalPromiseConstructor* constructor = new (NotNull, allocateCell<JSInternalPromiseConstructor>(vm)) JSInternalPromiseConstructor(vm, executable, globalObject, structure);
    constructor->finishCreation(vm, promisePrototype);
    return constructor;
}

Structure* JSInternalPromiseConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(JSFunctionType, StructureFlags), info());
}

JSInternalPromiseConstructor::JSInternalPromiseConstructor(VM& vm, FunctionExecutable* executable, JSGlobalObject* globalObject, Structure* structure)
    : Base(vm, executable, globalObject, structure)
{
}

} // namespace JSC
