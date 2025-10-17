/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 23, 2025.
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
#include "WindowOrWorkerGlobalScope.h"

#include "ExceptionOr.h"
#include "JSDOMExceptionHandling.h"
#include "JSDOMGlobalObject.h"
#include "MessagePort.h"
#include "SerializedScriptValue.h"
#include "StructuredSerializeOptions.h"
#include <JavaScriptCore/Exception.h>
#include <JavaScriptCore/JSCJSValueInlines.h>

namespace WebCore {

void WindowOrWorkerGlobalScope::reportError(JSDOMGlobalObject& globalObject, JSC::JSValue error)
{
    auto& vm = globalObject.vm();
    RELEASE_ASSERT(vm.currentThreadIsHoldingAPILock());
    auto* exception = JSC::jsDynamicCast<JSC::Exception*>(error);
    if (!exception)
        exception = JSC::Exception::create(vm, error);

    reportException(&globalObject, exception);
}

ExceptionOr<JSC::JSValue> WindowOrWorkerGlobalScope::structuredClone(JSDOMGlobalObject& lexicalGlobalObject, JSDOMGlobalObject& relevantGlobalObject, JSC::JSValue value, StructuredSerializeOptions&& options)
{
    Vector<Ref<MessagePort>> ports;
    auto messageData = SerializedScriptValue::create(lexicalGlobalObject, value, WTFMove(options.transfer), ports, SerializationForStorage::No, SerializationContext::WindowPostMessage);
    if (messageData.hasException())
        return messageData.releaseException();

    auto disentangledPorts = MessagePort::disentanglePorts(WTFMove(ports));
    if (disentangledPorts.hasException())
        return disentangledPorts.releaseException();

    Vector<Ref<MessagePort>> entangledPorts;
    if (auto* scriptExecutionContext = relevantGlobalObject.scriptExecutionContext())
        entangledPorts = MessagePort::entanglePorts(*scriptExecutionContext, disentangledPorts.releaseReturnValue());

    return messageData.returnValue()->deserialize(lexicalGlobalObject, &relevantGlobalObject, WTFMove(entangledPorts));
}

} // namespace WebCore
