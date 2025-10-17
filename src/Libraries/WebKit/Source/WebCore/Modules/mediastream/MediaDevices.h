/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 18, 2024.
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

#if ENABLE(MEDIA_STREAM)

#include "ActiveDOMObject.h"
#include "EventNames.h"
#include "EventTarget.h"
#include "ExceptionOr.h"
#include "IDLTypes.h"
#include "MediaTrackConstraints.h"
#include "RealtimeMediaSourceCenter.h"
#include "UserMediaClient.h"
#include <wtf/RobinHoodHashMap.h>
#include <wtf/RunLoop.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class Document;
class InputDeviceInfo;
class MediaDeviceInfo;
class MediaStream;
class UserGestureToken;

struct MediaTrackSupportedConstraints;

template<typename IDLType> class DOMPromiseDeferred;

class MediaDevices final : public RefCounted<MediaDevices>, public ActiveDOMObject, public EventTarget {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MediaDevices);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<MediaDevices> create(Document&);

    ~MediaDevices();

    Document* document() const;

    using Promise = DOMPromiseDeferred<IDLInterface<MediaStream>>;
    using EnumerateDevicesPromise = DOMPromiseDeferred<IDLSequence<IDLUnion<IDLInterface<MediaDeviceInfo>, IDLInterface<InputDeviceInfo>>>>;

    enum class DisplayCaptureSurfaceType {
        Monitor,
        Window,
        Application,
        Browser,
    };

    struct StreamConstraints {
        std::variant<bool, MediaTrackConstraints> video;
        std::variant<bool, MediaTrackConstraints> audio;
    };
    void getUserMedia(StreamConstraints&&, Promise&&);

    struct DisplayMediaStreamConstraints {
        std::variant<bool, MediaTrackConstraints> video;
        std::variant<bool, MediaTrackConstraints> audio;
    };
    void getDisplayMedia(DisplayMediaStreamConstraints&&, Promise&&);

    void enumerateDevices(EnumerateDevicesPromise&&);
    MediaTrackSupportedConstraints getSupportedConstraints();

    String deviceIdToPersistentId(const String& deviceId) const { return m_audioOutputDeviceIdToPersistentId.get(deviceId); }

    void willStartMediaCapture(bool microphone, bool camera);

private:
    explicit MediaDevices(Document&);

    void scheduledEventTimerFired();
    bool addEventListener(const AtomString& eventType, Ref<EventListener>&&, const AddEventListenerOptions&) override;

    void exposeDevices(Vector<CaptureDeviceWithCapabilities>&&, MediaDeviceHashSalts&&, EnumerateDevicesPromise&&);
    void listenForDeviceChanges();

    friend class JSMediaDevicesOwner;

    // ActiveDOMObject
    void stop() final;
    bool virtualHasPendingActivity() const final;

    // EventTarget.
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::MediaDevices; }
    ScriptExecutionContext* scriptExecutionContext() const final { return ActiveDOMObject::scriptExecutionContext(); }
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    enum class GestureAllowedRequest {
        Microphone = 1 << 0,
        Camera = 1 << 1,
        Display = 1 << 2,
    };
    bool computeUserGesturePriviledge(GestureAllowedRequest);

    RunLoop::Timer m_scheduledEventTimer;
    Markable<UserMediaClient::DeviceChangeObserverToken> m_deviceChangeToken;
    const EventNames& m_eventNames; // Need to cache this so we can use it from GC threads.
    bool m_listeningForDeviceChanges { false };

    OptionSet<GestureAllowedRequest> m_requestTypesForCurrentGesture;
    WeakPtr<UserGestureToken> m_currentGestureToken;

    MemoryCompactRobinHoodHashMap<String, String> m_audioOutputDeviceIdToPersistentId;
    String m_audioOutputDeviceId;

    bool m_hasRestrictedCameraDevices { true };
    bool m_hasRestrictedMicrophoneDevices { true };
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
