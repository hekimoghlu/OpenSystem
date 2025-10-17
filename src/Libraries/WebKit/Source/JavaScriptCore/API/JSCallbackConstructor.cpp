/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 25, 2024.
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
#include "JSCallbackConstructor.h"

#include "APICallbackFunction.h"
#include "APICast.h"
#include "JSCInlines.h"
#include "JSLock.h"

namespace JSC {

const ClassInfo JSCallbackConstructor::s_info = { "CallbackConstructor"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSCallbackConstructor) };

JSCallbackConstructor::JSCallbackConstructor(JSGlobalObject* globalObject, Structure* structure, JSClassRef jsClass, JSObjectCallAsConstructorCallback callback)
    : Base(globalObject->vm(), structure)
    , m_class(jsClass)
    , m_callback(callback)
{
}

void JSCallbackConstructor::finishCreation(JSGlobalObject* globalObject, JSClassRef jsClass)
{
    Base::finishCreation(globalObject->vm());
    ASSERT(inherits(info()));
    if (m_class)
        JSClassRetain(jsClass);
}

JSCallbackConstructor::~JSCallbackConstructor()
{
    if (m_class)
        JSClassRelease(m_class);
}

void JSCallbackConstructor::destroy(JSCell* cell)
{
    static_cast<JSCallbackConstructor*>(cell)->JSCallbackConstructor::~JSCallbackConstructor();
}

static JSC_DECLARE_HOST_FUNCTION(constructJSCallbackConstructor);

JSC_DEFINE_HOST_FUNCTION(constructJSCallbackConstructor, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return APICallbackFunction::constructImpl<JSCallbackConstructor>(globalObject, callFrame);
}

CallData JSCallbackConstructor::getConstructData(JSCell*)
{
    CallData constructData;
    constructData.type = CallData::Type::Native;
    constructData.native.function = constructJSCallbackConstructor;
    constructData.native.isBoundFunction = false;
    constructData.native.isWasm = false;
    return constructData;
}

} // namespace JSC
