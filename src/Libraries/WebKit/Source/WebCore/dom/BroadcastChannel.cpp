/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 25, 2023.
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
#include "BroadcastChannel.h"

#include "BroadcastChannelRegistry.h"
#include "EventNames.h"
#include "MessageEvent.h"
#include "Page.h"
#include "PartitionedSecurityOrigin.h"
#include "SecurityOrigin.h"
#include "SerializedScriptValue.h"
#include "WorkerGlobalScope.h"
#include "WorkerLoaderProxy.h"
#include "WorkerThread.h"
#include <wtf/CallbackAggregator.h>
#include <wtf/HashMap.h>
#include <wtf/Identified.h>
#include <wtf/MainThread.h>
#include <wtf/Scope.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(BroadcastChannel);

static Lock allBroadcastChannelsLock;
static HashMap<BroadcastChannelIdentifier, ThreadSafeWeakPtr<BroadcastChannel>>& allBroadcastChannels() WTF_REQUIRES_LOCK(allBroadcastChannelsLock)
{
    static NeverDestroyed<HashMap<BroadcastChannelIdentifier, ThreadSafeWeakPtr<BroadcastChannel>>> map;
    return map;
}

static HashMap<BroadcastChannelIdentifier, ScriptExecutionContextIdentifier>& channelToContextIdentifier()
{
    ASSERT(isMainThread());
    static NeverDestroyed<HashMap<BroadcastChannelIdentifier, ScriptExecutionContextIdentifier>> map;
    return map;
}

static PartitionedSecurityOrigin partitionedSecurityOriginFromContext(ScriptExecutionContext& context)
{
    Ref securityOrigin { *context.securityOrigin() };
    Ref topOrigin { context.settingsValues().broadcastChannelOriginPartitioningEnabled ? context.topOrigin() : securityOrigin.get() };
    return { WTFMove(topOrigin), WTFMove(securityOrigin) };
}

class BroadcastChannel::MainThreadBridge : public ThreadSafeRefCounted<MainThreadBridge, WTF::DestructionThread::Main>, public Identified<BroadcastChannelIdentifier> {
public:
    static Ref<MainThreadBridge> create(BroadcastChannel& channel, const String& name)
    {
        return adoptRef(*new MainThreadBridge(channel, name));
    }

    void registerChannel();
    void unregisterChannel();
    void postMessage(Ref<SerializedScriptValue>&&);
    void detach() { m_broadcastChannel = nullptr; }

    String name() const { return m_name.isolatedCopy(); }

private:
    MainThreadBridge(BroadcastChannel&, const String& name);

    void ensureOnMainThread(Function<void(Page*)>&&);

    WeakPtr<BroadcastChannel, WeakPtrImplWithEventTargetData> m_broadcastChannel;
    const String m_name; // Main thread only.
    PartitionedSecurityOrigin m_origin; // Main thread only.
};

BroadcastChannel::MainThreadBridge::MainThreadBridge(BroadcastChannel& channel, const String& name)
    : m_broadcastChannel(channel)
    , m_name(name.isolatedCopy())
    , m_origin(partitionedSecurityOriginFromContext(*channel.protectedScriptExecutionContext()).isolatedCopy())
{
}

void BroadcastChannel::MainThreadBridge::ensureOnMainThread(Function<void(Page*)>&& task)
{
    ASSERT(m_broadcastChannel);
    if (!m_broadcastChannel)
        return;

    RefPtr context = m_broadcastChannel->scriptExecutionContext();
    if (!context)
        return;
    ASSERT(context->isContextThread());

    if (auto* document = dynamicDowncast<Document>(*context)) {
        task(document->protectedPage().get());
        return;
    }

    auto* workerLoaderProxy = downcast<WorkerGlobalScope>(*context).thread().workerLoaderProxy();
    if (!workerLoaderProxy)
        return;

    workerLoaderProxy->postTaskToLoader([task = WTFMove(task)](auto& context) {
        task(downcast<Document>(context).protectedPage().get());
    });
}

void BroadcastChannel::MainThreadBridge::registerChannel()
{
    ensureOnMainThread([this, contextIdentifier = m_broadcastChannel->scriptExecutionContext()->identifier()](auto* page) mutable {
        if (page)
            page->protectedBroadcastChannelRegistry()->registerChannel(m_origin, m_name, identifier());
        channelToContextIdentifier().add(identifier(), contextIdentifier);
    });
}

void BroadcastChannel::MainThreadBridge::unregisterChannel()
{
    ensureOnMainThread([this](auto* page) {
        if (page)
            page->protectedBroadcastChannelRegistry()->unregisterChannel(m_origin, m_name, identifier());
        channelToContextIdentifier().remove(identifier());
    });
}

void BroadcastChannel::MainThreadBridge::postMessage(Ref<SerializedScriptValue>&& message)
{
    ensureOnMainThread([this, message = WTFMove(message)](auto* page) mutable {
        if (!page)
            return;

        auto blobHandles = message->blobHandles();
        page->protectedBroadcastChannelRegistry()->postMessage(m_origin, m_name, identifier(), WTFMove(message), [blobHandles = WTFMove(blobHandles)] {
            // Keeps Blob data inside messageData alive until the message has been delivered.
        });
    });
}

BroadcastChannel::BroadcastChannel(ScriptExecutionContext& context, const String& name)
    : ActiveDOMObject(&context)
    , m_mainThreadBridge(MainThreadBridge::create(*this, name))
{
    Ref mainThreadBridge = m_mainThreadBridge;
    {
        Locker locker { allBroadcastChannelsLock };
        allBroadcastChannels().add(mainThreadBridge->identifier(), *this);
    }
    mainThreadBridge->registerChannel();
}

BroadcastChannel::~BroadcastChannel()
{
    close();
    m_mainThreadBridge->detach();
    {
        Locker locker { allBroadcastChannelsLock };
        allBroadcastChannels().remove(m_mainThreadBridge->identifier());
    }
}

auto BroadcastChannel::protectedMainThreadBridge() const -> Ref<MainThreadBridge>
{
    return m_mainThreadBridge;
}

BroadcastChannelIdentifier BroadcastChannel::identifier() const
{
    return m_mainThreadBridge->identifier();
}

String BroadcastChannel::name() const
{
    return m_mainThreadBridge->name();
}

ExceptionOr<void> BroadcastChannel::postMessage(JSC::JSGlobalObject& globalObject, JSC::JSValue message)
{
    if (!isEligibleForMessaging())
        return { };

    if (m_isClosed)
        return Exception { ExceptionCode::InvalidStateError, "This BroadcastChannel is closed"_s };

    Vector<Ref<MessagePort>> ports;
    auto messageData = SerializedScriptValue::create(globalObject, message, { }, ports, SerializationForStorage::No, SerializationContext::WorkerPostMessage);
    if (messageData.hasException())
        return messageData.releaseException();
    ASSERT(ports.isEmpty());

    protectedMainThreadBridge()->postMessage(messageData.releaseReturnValue());
    return { };
}

void BroadcastChannel::close()
{
    if (m_isClosed)
        return;

    m_isClosed = true;
    protectedMainThreadBridge()->unregisterChannel();
}

void BroadcastChannel::dispatchMessageTo(BroadcastChannelIdentifier channelIdentifier, Ref<SerializedScriptValue>&& message, CompletionHandler<void()>&& completionHandler)
{
    ASSERT(isMainThread());
    auto completionHandlerCallingScope = makeScopeExit([completionHandler = WTFMove(completionHandler)]() mutable {
        callOnMainThread(WTFMove(completionHandler));
    });

    auto contextIdentifier = channelToContextIdentifier().get(channelIdentifier);
    if (!contextIdentifier)
        return;

    ScriptExecutionContext::ensureOnContextThread(contextIdentifier, [channelIdentifier, message = WTFMove(message), completionHandlerCallingScope = WTFMove(completionHandlerCallingScope)](auto&) mutable {
        RefPtr<BroadcastChannel> channel;
        {
            Locker locker { allBroadcastChannelsLock };
            channel = allBroadcastChannels().get(channelIdentifier).get();
        }
        if (channel)
            channel->dispatchMessage(WTFMove(message));
    });
}

void BroadcastChannel::dispatchMessage(Ref<SerializedScriptValue>&& message)
{
    if (!isEligibleForMessaging())
        return;

    if (m_isClosed)
        return;

    queueTaskKeepingObjectAlive(*this, TaskSource::PostedMessageQueue, [this, message = WTFMove(message)]() mutable {
        if (m_isClosed || !scriptExecutionContext())
            return;

        auto* globalObject = scriptExecutionContext()->globalObject();
        if (!globalObject)
            return;

        auto& vm = globalObject->vm();
        auto scope = DECLARE_CATCH_SCOPE(vm);
        auto event = MessageEvent::create(*globalObject, WTFMove(message), scriptExecutionContext()->securityOrigin()->toString());
        if (UNLIKELY(scope.exception())) {
            // Currently, we assume that the only way we can get here is if we have a termination.
            RELEASE_ASSERT(vm.hasPendingTerminationException());
            return;
        }

        dispatchEvent(event.event);
    });
}

void BroadcastChannel::eventListenersDidChange()
{
    m_hasRelevantEventListener = hasEventListeners(eventNames().messageEvent);
}

bool BroadcastChannel::virtualHasPendingActivity() const
{
    return !m_isClosed && m_hasRelevantEventListener;
}

// https://html.spec.whatwg.org/#eligible-for-messaging
bool BroadcastChannel::isEligibleForMessaging() const
{
    RefPtr context = scriptExecutionContext();
    if (!context)
        return false;

    if (auto document = dynamicDowncast<Document>(*context))
        return document->isFullyActive();

    return !downcast<WorkerGlobalScope>(*context).isClosing();
}

} // namespace WebCore
