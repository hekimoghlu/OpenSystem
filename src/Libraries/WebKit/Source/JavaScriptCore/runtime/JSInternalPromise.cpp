/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 18, 2025.
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
#include "JSInternalPromise.h"

#include "BuiltinNames.h"
#include "JSCInlines.h"

namespace JSC {

const ClassInfo JSInternalPromise::s_info = { "InternalPromise"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSInternalPromise) };

JSInternalPromise* JSInternalPromise::create(VM& vm, Structure* structure)
{
    JSInternalPromise* promise = new (NotNull, allocateCell<JSInternalPromise>(vm)) JSInternalPromise(vm, structure);
    promise->finishCreation(vm);
    return promise;
}

JSInternalPromise* JSInternalPromise::createWithInitialValues(VM& vm, Structure* structure)
{
    return create(vm, structure);
}

Structure* JSInternalPromise::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(JSPromiseType, StructureFlags), info());
}

JSInternalPromise::JSInternalPromise(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

JSInternalPromise* JSInternalPromise::then(JSGlobalObject* globalObject, JSFunction* onFulfilled, JSFunction* onRejected)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSObject* function = getAs<JSObject*>(globalObject, vm.propertyNames->builtinNames().thenPublicName());
    RETURN_IF_EXCEPTION(scope, nullptr);
    auto callData = JSC::getCallData(function);
    ASSERT(callData.type != CallData::Type::None);

    MarkedArgumentBuffer arguments;
    arguments.append(onFulfilled ? onFulfilled : jsUndefined());
    arguments.append(onRejected ? onRejected : jsUndefined());
    ASSERT(!arguments.hasOverflowed());
    JSValue result = call(globalObject, function, callData, this, arguments);
    RETURN_IF_EXCEPTION(scope, nullptr);
    return jsCast<JSInternalPromise*>(result);
}

JSInternalPromise* JSInternalPromise::rejectWithCaughtException(JSGlobalObject* globalObject, ThrowScope& scope)
{
    return jsCast<JSInternalPromise*>(JSPromise::rejectWithCaughtException(globalObject, scope));
}

} // namespace JSC
