/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 30, 2022.
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
#include "ServiceWorkerClient.h"

#include "MessagePort.h"
#include "SWContextManager.h"
#include "ScriptExecutionContext.h"
#include "SerializedScriptValue.h"
#include "ServiceWorkerGlobalScope.h"
#include "ServiceWorkerThread.h"
#include "ServiceWorkerWindowClient.h"
#include "StructuredSerializeOptions.h"

namespace WebCore {

Ref<ServiceWorkerClient> ServiceWorkerClient::create(ServiceWorkerGlobalScope& context, ServiceWorkerClientData&& data)
{
    if (data.type == ServiceWorkerClientType::Window)
        return ServiceWorkerWindowClient::create(context, WTFMove(data));

    return adoptRef(*new ServiceWorkerClient { context, WTFMove(data) });
}

ServiceWorkerClient::ServiceWorkerClient(ServiceWorkerGlobalScope& context, ServiceWorkerClientData&& data)
    : ContextDestructionObserver(&context)
    , m_data(WTFMove(data))
{
}

ServiceWorkerClient::~ServiceWorkerClient()
{
}

const URL& ServiceWorkerClient::url() const
{
    return m_data.url;
}

auto ServiceWorkerClient::type() const -> Type
{
    return m_data.type;
}

auto ServiceWorkerClient::frameType() const -> FrameType
{
    return m_data.frameType;
}

String ServiceWorkerClient::id() const
{
    return identifier().toString();
}

ExceptionOr<void> ServiceWorkerClient::postMessage(JSC::JSGlobalObject& globalObject, JSC::JSValue messageValue, StructuredSerializeOptions&& options)
{
    Vector<Ref<MessagePort>> ports;
    auto messageData = SerializedScriptValue::create(globalObject, messageValue, WTFMove(options.transfer), ports, SerializationForStorage::No, SerializationContext::WorkerPostMessage);
    if (messageData.hasException())
        return messageData.releaseException();

    // Disentangle the port in preparation for sending it to the remote context.
    auto portsOrException = MessagePort::disentanglePorts(WTFMove(ports));
    if (portsOrException.hasException())
        return portsOrException.releaseException();

    MessageWithMessagePorts message = { messageData.releaseReturnValue(), portsOrException.releaseReturnValue() };
    auto& context = downcast<ServiceWorkerGlobalScope>(*scriptExecutionContext());
    auto sourceIdentifier = context.thread().identifier();
    callOnMainThread([message = WTFMove(message), destinationIdentifier = identifier(), sourceIdentifier, sourceOrigin = context.origin().isolatedCopy()] {
        if (auto* connection = SWContextManager::singleton().connection())
            connection->postMessageToServiceWorkerClient(destinationIdentifier, message, sourceIdentifier, sourceOrigin);
    });

    return { };
}

} // namespace WebCore
