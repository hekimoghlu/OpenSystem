/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 6, 2022.
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
#include "JSAsyncFunction.h"

#include "JSCellInlines.h"
#include "JSFunctionInlines.h"
#include "VM.h"

namespace JSC {

const ClassInfo JSAsyncFunction::s_info = { "AsyncFunction"_s,  &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSAsyncFunction) };

JSAsyncFunction::JSAsyncFunction(VM& vm, FunctionExecutable* executable, JSScope* scope, Structure* structure)
    : Base(vm, executable, scope, structure)
{
}

JSAsyncFunction* JSAsyncFunction::createImpl(VM& vm, FunctionExecutable* executable, JSScope* scope, Structure* structure)
{
    JSAsyncFunction* asyncFunction = new (NotNull, allocateCell<JSAsyncFunction>(vm)) JSAsyncFunction(vm, executable, scope, structure);
    ASSERT(asyncFunction->structure()->globalObject());
    asyncFunction->finishCreation(vm);
    return asyncFunction;
}

JSAsyncFunction* JSAsyncFunction::create(VM& vm, JSGlobalObject* globalObject, FunctionExecutable* executable, JSScope* scope)
{
    return create(vm, globalObject, executable, scope, globalObject->asyncFunctionStructure());
}

JSAsyncFunction* JSAsyncFunction::create(VM& vm, JSGlobalObject*, FunctionExecutable* executable, JSScope* scope, Structure* structure)
{
    JSAsyncFunction* asyncFunction = createImpl(vm, executable, scope, structure);
    executable->notifyCreation(vm, asyncFunction, "Allocating an async function");
    return asyncFunction;
}

JSAsyncFunction* JSAsyncFunction::createWithInvalidatedReallocationWatchpoint(VM& vm, JSGlobalObject* globalObject, FunctionExecutable* executable, JSScope* scope)
{
    return createWithInvalidatedReallocationWatchpoint(vm, globalObject, executable, scope, globalObject->asyncFunctionStructure());
}

JSAsyncFunction* JSAsyncFunction::createWithInvalidatedReallocationWatchpoint(VM& vm, JSGlobalObject*, FunctionExecutable* executable, JSScope* scope, Structure* structure)
{
    return createImpl(vm, executable, scope, structure);
}

}
