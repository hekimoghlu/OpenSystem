/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 7, 2023.
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

#if ENABLE(WIRELESS_PLAYBACK_TARGET)

#include "ActiveDOMObject.h"
#include "EventTarget.h"
#include "WebCoreOpaqueRoot.h"
#include <wtf/HashMap.h>
#include <wtf/LoggerHelper.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class DeferredPromise;
class HTMLMediaElement;
class MediaPlaybackTarget;
class Node;
class RemotePlaybackAvailabilityCallback;

class RemotePlayback final
    : public RefCounted<RemotePlayback>
    , public ActiveDOMObject
    , public EventTarget
{
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RemotePlayback);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<RemotePlayback> create(HTMLMediaElement&);
    ~RemotePlayback();

    void watchAvailability(Ref<RemotePlaybackAvailabilityCallback>&&, Ref<DeferredPromise>&&);
    void cancelWatchAvailability(std::optional<int32_t> id, Ref<DeferredPromise>&&);
    void prompt(Ref<DeferredPromise>&&);

    bool hasAvailabilityCallbacks() const;
    void availabilityChanged(bool);
    void playbackTargetPickerWasDismissed();
    void shouldPlayToRemoteTargetChanged(bool);
    void isPlayingToRemoteTargetChanged(bool);

    enum class State {
        Connecting,
        Connected,
        Disconnected,
    };
    State state() const { return m_state; }

    void invalidate();

    WebCoreOpaqueRoot opaqueRootConcurrently() const;
    Node* ownerNode() const;

private:
    explicit RemotePlayback(HTMLMediaElement&);

    void setState(State);
    void establishConnection();
    void disconnect();

    // EventTarget.
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::RemotePlayback; }
    ScriptExecutionContext* scriptExecutionContext() const final { return ActiveDOMObject::scriptExecutionContext(); }
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    // ActiveDOMObject
    void stop() final;

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const { return m_logger.get(); }
    uint64_t logIdentifier() const { return m_logIdentifier; }
    WTFLogChannel& logChannel() const;
    ASCIILiteral logClassName() const { return "RemotePlayback"_s; }

    Ref<const Logger> m_logger;
    uint64_t m_logIdentifier { 0 };
#endif

    WeakPtr<HTMLMediaElement> m_mediaElement;
    uint32_t m_nextId { 0 };

    using CallbackMap = HashMap<int32_t, Ref<RemotePlaybackAvailabilityCallback>>;
    CallbackMap m_callbackMap;

    using PromiseVector = Vector<Ref<DeferredPromise>>;
    PromiseVector m_availabilityPromises;
    PromiseVector m_cancelAvailabilityPromises;
    PromiseVector m_promptPromises;
    State m_state { State::Disconnected };
    bool m_available { false };
};

}

#endif // ENABLE(WIRELESS_PLAYBACK_TARGET)

