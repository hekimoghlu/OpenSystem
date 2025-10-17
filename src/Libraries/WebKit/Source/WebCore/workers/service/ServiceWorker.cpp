/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 23, 2024.
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
#include "ServiceWorker.h"

#include "Document.h"
#include "EventNames.h"
#include "Logging.h"
#include "MessagePort.h"
#include "SWClientConnection.h"
#include "ScriptExecutionContext.h"
#include "SerializedScriptValue.h"
#include "ServiceWorkerClientData.h"
#include "ServiceWorkerContainer.h"
#include "ServiceWorkerGlobalScope.h"
#include "ServiceWorkerProvider.h"
#include "ServiceWorkerThread.h"
#include "StructuredSerializeOptions.h"
#include "WorkerSWClientConnection.h"
#include <JavaScriptCore/JSCJSValueInlines.h>
#include <wtf/EnumTraits.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

#define WORKER_RELEASE_LOG(fmt, ...) RELEASE_LOG(ServiceWorker, "%p - ServiceWorker::" fmt, this, ##__VA_ARGS__)
#define WORKER_RELEASE_LOG_ERROR(fmt, ...) RELEASE_LOG_ERROR(ServiceWorker, "%p - ServiceWorker::" fmt, this, ##__VA_ARGS__)

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ServiceWorker);

Ref<ServiceWorker> ServiceWorker::getOrCreate(ScriptExecutionContext& context, ServiceWorkerData&& data)
{
    if (auto existingServiceWorker = context.serviceWorker(data.identifier))
        return *existingServiceWorker;
    auto serviceWorker = adoptRef(*new ServiceWorker(context, WTFMove(data)));
    serviceWorker->suspendIfNeeded();
    return serviceWorker;
}

ServiceWorker::ServiceWorker(ScriptExecutionContext& context, ServiceWorkerData&& data)
    : ActiveDOMObject(&context)
    , m_data(WTFMove(data))
{
    context.registerServiceWorker(*this);

    relaxAdoptionRequirement();
    updatePendingActivityForEventDispatch();

    WORKER_RELEASE_LOG("serviceWorkerID=%" PRIu64 ", state=%hhu", identifier().toUInt64(), enumToUnderlyingType(m_data.state));
}

ServiceWorker::~ServiceWorker()
{
    if (auto* context = scriptExecutionContext())
        context->unregisterServiceWorker(*this);
}

void ServiceWorker::updateState(State state)
{
    WORKER_RELEASE_LOG("updateState: Updating service worker %" PRIu64 " state from %hhu to %hhu. registrationID=%" PRIu64, identifier().toUInt64(), enumToUnderlyingType(m_data.state), enumToUnderlyingType(state), registrationIdentifier().toUInt64());
    m_data.state = state;
    if (state != State::Installing && !m_isStopped) {
        ASSERT(m_pendingActivityForEventDispatch);
        dispatchEvent(Event::create(eventNames().statechangeEvent, Event::CanBubble::No, Event::IsCancelable::No));
    }

    updatePendingActivityForEventDispatch();
}

SWClientConnection& ServiceWorker::swConnection()
{
    ASSERT(scriptExecutionContext());
    if (auto* worker = dynamicDowncast<WorkerGlobalScope>(scriptExecutionContext()))
        return worker->swClientConnection();
    return ServiceWorkerProvider::singleton().serviceWorkerConnection();
}

ExceptionOr<void> ServiceWorker::postMessage(JSC::JSGlobalObject& globalObject, JSC::JSValue messageValue, StructuredSerializeOptions&& options)
{
    if (m_isStopped)
        return Exception { ExceptionCode::InvalidStateError };

    Vector<Ref<MessagePort>> ports;
    auto messageData = SerializedScriptValue::create(globalObject, messageValue, WTFMove(options.transfer), ports, SerializationForStorage::No, SerializationContext::WorkerPostMessage);
    if (messageData.hasException())
        return messageData.releaseException();

    // Disentangle the port in preparation for sending it to the remote context.
    auto portsOrException = MessagePort::disentanglePorts(WTFMove(ports));
    if (portsOrException.hasException())
        return portsOrException.releaseException();

    auto& context = *scriptExecutionContext();
    // FIXME: Maybe we could use a ScriptExecutionContextIdentifier for service workers too.
    ServiceWorkerOrClientIdentifier sourceIdentifier = [&]() -> ServiceWorkerOrClientIdentifier {
        if (auto* serviceWorker = dynamicDowncast<ServiceWorkerGlobalScope>(context))
            return serviceWorker->thread().identifier();
        return context.identifier();
    }();

    MessageWithMessagePorts message { messageData.releaseReturnValue(), portsOrException.releaseReturnValue() };
    swConnection().postMessageToServiceWorker(identifier(), WTFMove(message), sourceIdentifier);
    return { };
}

enum EventTargetInterfaceType ServiceWorker::eventTargetInterface() const
{
    return EventTargetInterfaceType::ServiceWorker;
}

ScriptExecutionContext* ServiceWorker::scriptExecutionContext() const
{
    return ContextDestructionObserver::scriptExecutionContext();
}

void ServiceWorker::stop()
{
    m_isStopped = true;
    removeAllEventListeners();
    scriptExecutionContext()->unregisterServiceWorker(*this);
    updatePendingActivityForEventDispatch();
}

void ServiceWorker::updatePendingActivityForEventDispatch()
{
    // ServiceWorkers can dispatch events until they become redundant or they are stopped.
    if (m_isStopped || state() == State::Redundant) {
        m_pendingActivityForEventDispatch = nullptr;
        return;
    }
    if (m_pendingActivityForEventDispatch)
        return;
    m_pendingActivityForEventDispatch = makePendingActivity(*this);
}

} // namespace WebCore
