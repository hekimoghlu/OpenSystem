/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 7, 2023.
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
#include "AsyncFunctionConstructor.h"

#include "AsyncFunctionPrototype.h"
#include "FunctionConstructor.h"
#include "JSCInlines.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(AsyncFunctionConstructor);

const ClassInfo AsyncFunctionConstructor::s_info = { "AsyncFunction"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(AsyncFunctionConstructor) };

static JSC_DECLARE_HOST_FUNCTION(callAsyncFunctionConstructor);
static JSC_DECLARE_HOST_FUNCTION(constructAsyncFunctionConstructor);

JSC_DEFINE_HOST_FUNCTION(callAsyncFunctionConstructor, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    ArgList args(callFrame);
    return JSValue::encode(constructFunction(globalObject, callFrame, args, FunctionConstructionMode::Async));
}

JSC_DEFINE_HOST_FUNCTION(constructAsyncFunctionConstructor, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    ArgList args(callFrame);
    return JSValue::encode(constructFunction(globalObject, callFrame, args, FunctionConstructionMode::Async, callFrame->newTarget()));
}

AsyncFunctionConstructor::AsyncFunctionConstructor(VM& vm, Structure* structure)
    : InternalFunction(vm, structure, callAsyncFunctionConstructor, constructAsyncFunctionConstructor)
{
}

void AsyncFunctionConstructor::finishCreation(VM& vm, AsyncFunctionPrototype* prototype)
{
    Base::finishCreation(vm, 1, "AsyncFunction"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, prototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
}

} // namespace JSC
