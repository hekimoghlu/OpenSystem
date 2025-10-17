/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 27, 2025.
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
#include "DedicatedWorkerGlobalScope.h"

#include "ContentSecurityPolicyResponseHeaders.h"
#include "DedicatedWorkerThread.h"
#include "EventNames.h"
#include "JSRTCRtpScriptTransformer.h"
#include "LocalDOMWindow.h"
#include "MessageEvent.h"
#include "RTCTransformEvent.h"
#include "RequestAnimationFrameCallback.h"
#include "SecurityOrigin.h"
#include "StructuredSerializeOptions.h"
#include "Worker.h"
#include "WorkerObjectProxy.h"
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(NOTIFICATIONS)
#include "WorkerNotificationClient.h"
#endif

#if ENABLE(OFFSCREEN_CANVAS)
#include "WorkerAnimationController.h"
#endif

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(DedicatedWorkerGlobalScope);

Ref<DedicatedWorkerGlobalScope> DedicatedWorkerGlobalScope::create(const WorkerParameters& params, Ref<SecurityOrigin>&& origin, DedicatedWorkerThread& thread, Ref<SecurityOrigin>&& topOrigin, IDBClient::IDBConnectionProxy* connectionProxy, SocketProvider* socketProvider, std::unique_ptr<WorkerClient>&& workerClient)
{
    auto context = adoptRef(*new DedicatedWorkerGlobalScope(params, WTFMove(origin), thread, WTFMove(topOrigin), connectionProxy, socketProvider, WTFMove(workerClient)));
    context->addToContextsMap();
    if (!params.shouldBypassMainWorldContentSecurityPolicy)
        context->applyContentSecurityPolicyResponseHeaders(params.contentSecurityPolicyResponseHeaders);
    return context;
}

DedicatedWorkerGlobalScope::DedicatedWorkerGlobalScope(const WorkerParameters& params, Ref<SecurityOrigin>&& origin, DedicatedWorkerThread& thread, Ref<SecurityOrigin>&& topOrigin, IDBClient::IDBConnectionProxy* connectionProxy, SocketProvider* socketProvider, std::unique_ptr<WorkerClient>&& workerClient)
    : WorkerGlobalScope(WorkerThreadType::DedicatedWorker, params, WTFMove(origin), thread, WTFMove(topOrigin), connectionProxy, socketProvider, WTFMove(workerClient))
    , m_name(params.name)
{
}

DedicatedWorkerGlobalScope::~DedicatedWorkerGlobalScope()
{
    // We need to remove from the contexts map very early in the destructor so that calling postTask() on this WorkerGlobalScope from another thread is safe.
    removeFromContextsMap();
}

enum EventTargetInterfaceType DedicatedWorkerGlobalScope::eventTargetInterface() const
{
    return EventTargetInterfaceType::DedicatedWorkerGlobalScope;
}

void DedicatedWorkerGlobalScope::prepareForDestruction()
{
    WorkerGlobalScope::prepareForDestruction();
}

ExceptionOr<void> DedicatedWorkerGlobalScope::postMessage(JSC::JSGlobalObject& state, JSC::JSValue messageValue, StructuredSerializeOptions&& options)
{
    Vector<Ref<MessagePort>> ports;

    auto message = SerializedScriptValue::create(state, messageValue, WTFMove(options.transfer), ports, SerializationForStorage::No, SerializationContext::WorkerPostMessage);
    if (message.hasException())
        return message.releaseException();

    // Disentangle the port in preparation for sending it to the remote context.
    auto channels = MessagePort::disentanglePorts(WTFMove(ports));
    if (channels.hasException())
        return channels.releaseException();

    thread().workerObjectProxy().postMessageToWorkerObject({ message.releaseReturnValue(), channels.releaseReturnValue() });
    return { };
}

DedicatedWorkerThread& DedicatedWorkerGlobalScope::thread()
{
    return static_cast<DedicatedWorkerThread&>(Base::thread());
}

#if ENABLE(OFFSCREEN_CANVAS_IN_WORKERS)
CallbackId DedicatedWorkerGlobalScope::requestAnimationFrame(Ref<RequestAnimationFrameCallback>&& callback)
{
    if (!m_workerAnimationController)
        m_workerAnimationController = WorkerAnimationController::create(*this);
    return m_workerAnimationController->requestAnimationFrame(WTFMove(callback));
}

void DedicatedWorkerGlobalScope::cancelAnimationFrame(CallbackId callbackId)
{
    if (m_workerAnimationController)
        m_workerAnimationController->cancelAnimationFrame(callbackId);
}
#endif

#if ENABLE(WEB_RTC)
RefPtr<RTCRtpScriptTransformer> DedicatedWorkerGlobalScope::createRTCRtpScriptTransformer(MessageWithMessagePorts&& options)
{
    auto transformerOrException = RTCRtpScriptTransformer::create(*this, WTFMove(options));
    if (transformerOrException.hasException())
        return nullptr;
    auto transformer = transformerOrException.releaseReturnValue();
    dispatchEvent(RTCTransformEvent::create(eventNames().rtctransformEvent, transformer.copyRef(), Event::IsTrusted::Yes));
    return transformer;
}
#endif

#if ENABLE(NOTIFICATIONS)
NotificationClient* DedicatedWorkerGlobalScope::notificationClient()
{
    if (!m_notificationClient)
        m_notificationClient = WorkerNotificationClient::create(*this);
    return m_notificationClient.get();
}
#endif

} // namespace WebCore
