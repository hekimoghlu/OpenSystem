/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 6, 2025.
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
#include "DrawingAreaProxy.h"

#include "DrawingAreaMessages.h"
#include "DrawingAreaProxyMessages.h"
#include "WebPageProxy.h"
#include "WebProcessProxy.h"
#include <WebCore/ScrollView.h>
#include <wtf/TZoneMallocInlines.h>

#if PLATFORM(COCOA)
#include "MessageSenderInlines.h"
#include <wtf/MachSendRight.h>
#endif

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(DrawingAreaProxy);

DrawingAreaProxy::DrawingAreaProxy(DrawingAreaType type, WebPageProxy& webPageProxy, WebProcessProxy& webProcessProxy)
    : m_type(type)
    , m_webPageProxy(webPageProxy)
    , m_webProcessProxy(webProcessProxy)
    , m_size(webPageProxy.viewSize())
#if PLATFORM(MAC)
    , m_viewExposedRectChangedTimer(RunLoop::main(), this, &DrawingAreaProxy::viewExposedRectChangedTimerFired)
#endif
{
}

DrawingAreaProxy::~DrawingAreaProxy() = default;

RefPtr<WebPageProxy> DrawingAreaProxy::protectedWebPageProxy() const
{
    return m_webPageProxy.get();
}

Ref<WebProcessProxy> DrawingAreaProxy::protectedWebProcessProxy() const
{
    return m_webProcessProxy.get();
}

void DrawingAreaProxy::startReceivingMessages(WebProcessProxy& process)
{
    for (auto& name : messageReceiverNames())
        process.addMessageReceiver(name, identifier(), *this);
}

void DrawingAreaProxy::stopReceivingMessages(WebProcessProxy& process)
{
    for (auto& name : messageReceiverNames())
        process.removeMessageReceiver(name, identifier());
}

std::span<IPC::ReceiverName> DrawingAreaProxy::messageReceiverNames() const
{
    static std::array<IPC::ReceiverName, 1> name { Messages::DrawingAreaProxy::messageReceiverName() };
    return { name };
}

IPC::Connection* DrawingAreaProxy::messageSenderConnection() const
{
    return &m_webProcessProxy->connection();
}

bool DrawingAreaProxy::sendMessage(UniqueRef<IPC::Encoder>&& encoder, OptionSet<IPC::SendOption> sendOptions)
{
    return protectedWebProcessProxy()->sendMessage(WTFMove(encoder), sendOptions);
}

bool DrawingAreaProxy::sendMessageWithAsyncReply(UniqueRef<IPC::Encoder>&& encoder, AsyncReplyHandler handler, OptionSet<IPC::SendOption> sendOptions)
{
    return protectedWebProcessProxy()->sendMessage(WTFMove(encoder), sendOptions, WTFMove(handler));
}

uint64_t DrawingAreaProxy::messageSenderDestinationID() const
{
    return identifier().toUInt64();
}

DelegatedScrollingMode DrawingAreaProxy::delegatedScrollingMode() const
{
    return DelegatedScrollingMode::NotDelegated;
}

bool DrawingAreaProxy::setSize(const IntSize& size, const IntSize& scrollDelta)
{ 
    if (m_size == size && scrollDelta.isZero())
        return false;

    m_size = size;
    m_scrollOffset += scrollDelta;
    sizeDidChange();
    return true;
}

WebPageProxy* DrawingAreaProxy::page() const
{
    return m_webPageProxy.get();
}

#if PLATFORM(COCOA)
MachSendRight DrawingAreaProxy::createFence()
{
    return MachSendRight();
}
#endif

#if PLATFORM(MAC)
void DrawingAreaProxy::didChangeViewExposedRect()
{
    if (!protectedWebPageProxy()->hasRunningProcess())
        return;

    if (!m_viewExposedRectChangedTimer.isActive())
        m_viewExposedRectChangedTimer.startOneShot(0_s);
}

void DrawingAreaProxy::viewExposedRectChangedTimerFired()
{
    RefPtr webPageProxy = m_webPageProxy.get();
    if (!webPageProxy || !webPageProxy->hasRunningProcess())
        return;

    auto viewExposedRect = webPageProxy->viewExposedRect();
    if (viewExposedRect == m_lastSentViewExposedRect)
        return;

    send(Messages::DrawingArea::SetViewExposedRect(viewExposedRect));
    m_lastSentViewExposedRect = viewExposedRect;
}
#endif // PLATFORM(MAC)

} // namespace WebKit
