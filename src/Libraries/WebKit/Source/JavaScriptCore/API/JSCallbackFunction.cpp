/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 13, 2022.
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
#include "JSCallbackFunction.h"

#include "APICallbackFunction.h"
#include "JSCInlines.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(JSCallbackFunction);

const ClassInfo JSCallbackFunction::s_info = { "CallbackFunction"_s, &InternalFunction::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSCallbackFunction) };

static JSC_DECLARE_HOST_FUNCTION(callJSCallbackFunction);

JSC_DEFINE_HOST_FUNCTION(callJSCallbackFunction, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return APICallbackFunction::callImpl<JSCallbackFunction>(globalObject, callFrame);
}

JSCallbackFunction::JSCallbackFunction(VM& vm, Structure* structure, JSObjectCallAsFunctionCallback callback)
    : InternalFunction(vm, structure, callJSCallbackFunction, nullptr)
    , m_callback(callback)
{
}

void JSCallbackFunction::finishCreation(VM& vm, const String& name)
{
    Base::finishCreation(vm, 0, name);
    ASSERT(inherits(info()));
}

JSCallbackFunction* JSCallbackFunction::create(VM& vm, JSGlobalObject* globalObject, JSObjectCallAsFunctionCallback callback, const String& name)
{
    Structure* structure = globalObject->callbackFunctionStructure();
    JSCallbackFunction* function = new (NotNull, allocateCell<JSCallbackFunction>(vm)) JSCallbackFunction(vm, structure, callback);
    function->finishCreation(vm, name);
    return function;
}

} // namespace JSC
