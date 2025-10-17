/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 12, 2024.
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
#include "RemoteDOMWindow.h"

#include "FrameDestructionObserverInlines.h"
#include "FrameLoader.h"
#include "LocalDOMWindow.h"
#include "MessagePort.h"
#include "NavigationScheduler.h"
#include "Page.h"
#include "RemoteFrame.h"
#include "RemoteFrameClient.h"
#include "SecurityOrigin.h"
#include "SerializedScriptValue.h"
#include <JavaScriptCore/JSCJSValue.h>
#include <JavaScriptCore/JSCJSValueInlines.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RemoteDOMWindow);

RemoteDOMWindow::RemoteDOMWindow(RemoteFrame& frame, GlobalWindowIdentifier&& identifier)
    : DOMWindow(WTFMove(identifier), DOMWindowType::Remote)
    , m_frame(frame)
{
}

RemoteDOMWindow::~RemoteDOMWindow() = default;

WindowProxy* RemoteDOMWindow::self() const
{
    if (!m_frame)
        return nullptr;
    return &m_frame->windowProxy();
}

void RemoteDOMWindow::closePage()
{
    if (!m_frame)
        return;
    m_frame->client().closePage();
}

void RemoteDOMWindow::frameDetached()
{
    m_frame = nullptr;
}

void RemoteDOMWindow::focus(LocalDOMWindow&)
{
    // FIXME(264713): Add security checks here equivalent to LocalDOMWindow::focus().
    if (m_frame && m_frame->isMainFrame())
        m_frame->client().focus();
}

void RemoteDOMWindow::blur()
{
    // FIXME(268121): Add security checks here equivalent to LocalDOMWindow::blur().
    if (m_frame && m_frame->isMainFrame())
        m_frame->client().unfocus();
}

unsigned RemoteDOMWindow::length() const
{
    if (!m_frame)
        return 0;

    return m_frame->tree().childCount();
}

ExceptionOr<void> RemoteDOMWindow::postMessage(JSC::JSGlobalObject& lexicalGlobalObject, LocalDOMWindow& incumbentWindow, JSC::JSValue message, WindowPostMessageOptions&& options)
{
    RefPtr sourceDocument = incumbentWindow.document();
    if (!sourceDocument)
        return { };

    RefPtr sourceFrame = incumbentWindow.frame();
    if (!sourceFrame)
        return { };

    auto targetSecurityOrigin = createTargetOriginForPostMessage(options.targetOrigin, *sourceDocument);
    if (targetSecurityOrigin.hasException())
        return targetSecurityOrigin.releaseException();

    std::optional<SecurityOriginData> target;
    if (auto origin = targetSecurityOrigin.releaseReturnValue())
        target = origin->data();

    Vector<Ref<MessagePort>> ports;
    auto messageData = SerializedScriptValue::create(lexicalGlobalObject, message, WTFMove(options.transfer), ports, SerializationForStorage::No, SerializationContext::WindowPostMessage);
    if (messageData.hasException())
        return messageData.releaseException();

    auto disentangledPorts = MessagePort::disentanglePorts(WTFMove(ports));
    if (disentangledPorts.hasException())
        return messageData.releaseException();

    // Capture the source of the message. We need to do this synchronously
    // in order to capture the source of the message correctly.
    auto sourceOrigin = sourceDocument->securityOrigin().toString();

    MessageWithMessagePorts messageWithPorts { messageData.releaseReturnValue(), disentangledPorts.releaseReturnValue() };
    if (auto* remoteFrame = frame())
        remoteFrame->client().postMessageToRemote(sourceFrame->frameID(), sourceOrigin, remoteFrame->frameID(), target, messageWithPorts);
    return { };
}

void RemoteDOMWindow::setLocation(LocalDOMWindow& activeWindow, const URL& completedURL, NavigationHistoryBehavior historyHandling, SetLocationLocking locking)
{
    // FIXME: Add some or all of the security checks in LocalDOMWindow::setLocation. <rdar://116500603>
    // FIXME: Refactor this duplicate code to share with LocalDOMWindow::setLocation. <rdar://116500603>

    RefPtr activeDocument = activeWindow.document();
    if (!activeDocument)
        return;

    RefPtr frame = this->frame();
    if (!activeDocument->canNavigate(frame.get(), completedURL))
        return;

    // We want a new history item if we are processing a user gesture.
    LockHistory lockHistory = (locking != SetLocationLocking::LockHistoryBasedOnGestureState || !UserGestureIndicator::processingUserGesture()) ? LockHistory::Yes : LockHistory::No;
    LockBackForwardList lockBackForwardList = (locking != SetLocationLocking::LockHistoryBasedOnGestureState) ? LockBackForwardList::Yes : LockBackForwardList::No;
    frame->protectedNavigationScheduler()->scheduleLocationChange(*activeDocument, activeDocument->securityOrigin(),
        // FIXME: What if activeDocument()->frame() is 0?
        completedURL, activeDocument->frame()->loader().outgoingReferrer(),
        lockHistory, lockBackForwardList,
        historyHandling);
}

} // namespace WebCore
