/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 26, 2025.
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
#pragma once

#include "ActiveDOMObject.h"
#include "EventLoop.h"
#include "EventTarget.h"
#include "ExceptionOr.h"
#include "ThreadableLoaderClient.h"
#include "Timer.h"
#include <wtf/URL.h>
#include <wtf/Vector.h>

namespace WebCore {

class MessageEvent;
class TextResourceDecoder;
class ThreadableLoader;

class EventSource final : public RefCounted<EventSource>, public EventTarget, private ThreadableLoaderClient, public ActiveDOMObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(EventSource);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(EventSource);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    struct Init {
        bool withCredentials;
    };
    static ExceptionOr<Ref<EventSource>> create(ScriptExecutionContext&, const String& url, const Init&);
    virtual ~EventSource();

    USING_CAN_MAKE_WEAKPTR(EventTarget);

    const String& url() const;
    bool withCredentials() const;

    using State = short;
    static const State CONNECTING = 0;
    static const State OPEN = 1;
    static const State CLOSED = 2;

    State readyState() const;

    void close();

private:
    EventSource(ScriptExecutionContext&, const URL&, const Init&);

    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::EventSource; }
    ScriptExecutionContext* scriptExecutionContext() const final { return ActiveDOMObject::scriptExecutionContext(); }

    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    void dispatchErrorEvent();
    void doExplicitLoadCancellation();

    // ThreadableLoaderClient
    void didReceiveResponse(ScriptExecutionContextIdentifier, std::optional<ResourceLoaderIdentifier>, const ResourceResponse&) final;
    void didReceiveData(const SharedBuffer&) final;
    void didFinishLoading(ScriptExecutionContextIdentifier, std::optional<ResourceLoaderIdentifier>, const NetworkLoadMetrics&) final;
    void didFail(std::optional<ScriptExecutionContextIdentifier>, const ResourceError&) final;

    // ActiveDOMObject
    void stop() final;
    void suspend(ReasonForSuspension) final;
    void resume() final;
    bool virtualHasPendingActivity() const final;

    void connect();
    void networkRequestEnded();
    void scheduleInitialConnect();
    void scheduleReconnect();
    void abortConnectionAttempt();
    void parseEventStream();
    void parseEventStreamLine(unsigned position, std::optional<unsigned> fieldLength, unsigned lineLength);
    void dispatchMessageEvent();

    bool responseIsValid(const ResourceResponse&) const;

    static const uint64_t defaultReconnectDelay;

    URL m_url;
    bool m_withCredentials;
    State m_state { CONNECTING };

    Ref<TextResourceDecoder> m_decoder;
    RefPtr<ThreadableLoader> m_loader;
    EventLoopTimerHandle m_connectTimer;
    Vector<UChar> m_receiveBuffer;
    bool m_discardTrailingNewline { false };
    bool m_requestInFlight { false };
    bool m_isSuspendedForBackForwardCache { false };
    bool m_isDoingExplicitCancellation { false };
    bool m_shouldReconnectOnResume { false };

    AtomString m_eventName;
    Vector<UChar> m_data;
    String m_currentlyParsedEventId;
    String m_lastEventId;
    uint64_t m_reconnectDelay { defaultReconnectDelay };
    String m_eventStreamOrigin;
};

inline const String& EventSource::url() const
{
    return m_url.string();
}

inline bool EventSource::withCredentials() const
{
    return m_withCredentials;
}

inline EventSource::State EventSource::readyState() const
{
    return m_state;
}

} // namespace WebCore
