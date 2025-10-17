/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 27, 2024.
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
#include "AsyncIteratorPrototype.h"

#include "JSCBuiltins.h"
#include "JSCInlines.h"

namespace JSC {

const ClassInfo AsyncIteratorPrototype::s_info = { "AsyncIterator"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(AsyncIteratorPrototype) };

static JSC_DECLARE_HOST_FUNCTION(asyncIteratorProtoFuncAsyncIterator);

void AsyncIteratorPrototype::finishCreation(VM& vm, JSGlobalObject* globalObject)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    JSFunction* asyncIteratorFunction = JSFunction::create(vm, globalObject, 0, "[Symbol.asyncIterator]"_s, asyncIteratorProtoFuncAsyncIterator, ImplementationVisibility::Public, AsyncIteratorIntrinsic);
    putDirectWithoutTransition(vm, vm.propertyNames->asyncIteratorSymbol, asyncIteratorFunction, static_cast<unsigned>(PropertyAttribute::DontEnum));
}

JSC_DEFINE_HOST_FUNCTION(asyncIteratorProtoFuncAsyncIterator, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return JSValue::encode(callFrame->thisValue().toThis(globalObject, ECMAMode::strict()));
}

} // namespace JSC
