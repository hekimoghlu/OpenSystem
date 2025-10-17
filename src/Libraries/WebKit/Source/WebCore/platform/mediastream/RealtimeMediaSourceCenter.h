/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 8, 2021.
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

#include "ExceptionOr.h"
#include "MediaStreamRequest.h"
#include "RealtimeMediaSource.h"
#include <wtf/AbstractRefCountedAndCanMakeWeakPtr.h>
#include <wtf/Function.h>
#include <wtf/RefPtr.h>
#include <wtf/RunLoop.h>
#include <wtf/text/WTFString.h>

#if PLATFORM(COCOA)
#include <pal/spi/cocoa/TCCSPI.h>
#include <wtf/OSObjectPtr.h>
#endif

namespace WebCore {

class CaptureDevice;
class CaptureDeviceManager;
class RealtimeMediaSourceSettings;
class TrackSourceInfo;

struct MediaConstraints;

class WEBCORE_EXPORT RealtimeMediaSourceCenterObserver : public AbstractRefCountedAndCanMakeWeakPtr<RealtimeMediaSourceCenterObserver> {
public:
    virtual ~RealtimeMediaSourceCenterObserver();

    virtual void devicesChanged() = 0;
    virtual void deviceWillBeRemoved(const String& persistentId) = 0;
};

class WEBCORE_EXPORT RealtimeMediaSourceCenter : public ThreadSafeRefCounted<RealtimeMediaSourceCenter, WTF::DestructionThread::MainRunLoop> {
public:
    ~RealtimeMediaSourceCenter();

    WEBCORE_EXPORT static RealtimeMediaSourceCenter& singleton();

    using ValidConstraintsHandler = Function<void(Vector<CaptureDevice>&& audioDeviceUIDs, Vector<CaptureDevice>&& videoDeviceUIDs)>;
    using InvalidConstraintsHandler = Function<void(MediaConstraintType)>;
    WEBCORE_EXPORT void validateRequestConstraints(ValidConstraintsHandler&&, InvalidConstraintsHandler&&, const MediaStreamRequest&, MediaDeviceHashSalts&&);

    using NewMediaStreamHandler = Function<void(Expected<Ref<MediaStreamPrivate>, CaptureSourceError>&&)>;
    void createMediaStream(Ref<const Logger>&&, NewMediaStreamHandler&&, MediaDeviceHashSalts&&, CaptureDevice&& audioDevice, CaptureDevice&& videoDevice, const MediaStreamRequest&);

    WEBCORE_EXPORT void getMediaStreamDevices(CompletionHandler<void(Vector<CaptureDevice>&&)>&&);
    WEBCORE_EXPORT std::optional<RealtimeMediaSourceCapabilities> getCapabilities(const CaptureDevice&);

    WEBCORE_EXPORT AudioCaptureFactory& audioCaptureFactory();
    WEBCORE_EXPORT void setAudioCaptureFactory(AudioCaptureFactory&);
    WEBCORE_EXPORT void unsetAudioCaptureFactory(AudioCaptureFactory&);

    WEBCORE_EXPORT VideoCaptureFactory& videoCaptureFactory();
    WEBCORE_EXPORT void setVideoCaptureFactory(VideoCaptureFactory&);
    WEBCORE_EXPORT void unsetVideoCaptureFactory(VideoCaptureFactory&);

    WEBCORE_EXPORT DisplayCaptureFactory& displayCaptureFactory();
    WEBCORE_EXPORT void setDisplayCaptureFactory(DisplayCaptureFactory&);
    WEBCORE_EXPORT void unsetDisplayCaptureFactory(DisplayCaptureFactory&);

    WEBCORE_EXPORT static String hashStringWithSalt(const String& id, const String& hashSalt);

    WEBCORE_EXPORT void addDevicesChangedObserver(RealtimeMediaSourceCenterObserver&);
    WEBCORE_EXPORT void removeDevicesChangedObserver(RealtimeMediaSourceCenterObserver&);

    void captureDevicesChanged();
    void captureDeviceWillBeRemoved(const String& persistentId);

    WEBCORE_EXPORT static bool shouldInterruptAudioOnPageVisibilityChange();

#if ENABLE(APP_PRIVACY_REPORT)
    void setIdentity(OSObjectPtr<tcc_identity_t>&& identity) { m_identity = WTFMove(identity); }
    OSObjectPtr<tcc_identity_t> identity() const { return m_identity; }
    bool hasIdentity() const { return !!m_identity; }
#endif

#if ENABLE(EXTENSION_CAPABILITIES)
    const String& currentMediaEnvironment() const;
    void setCurrentMediaEnvironment(String&&);
#endif

private:
    RealtimeMediaSourceCenter();
    friend class NeverDestroyed<RealtimeMediaSourceCenter>;

    AudioCaptureFactory& defaultAudioCaptureFactory();
    VideoCaptureFactory& defaultVideoCaptureFactory();
    DisplayCaptureFactory& defaultDisplayCaptureFactory();

    struct DeviceInfo {
        double fitnessScore;
        CaptureDevice device;
    };

    void getDisplayMediaDevices(const MediaStreamRequest&, MediaDeviceHashSalts&&, Vector<DeviceInfo>&, MediaConstraintType&);
    void getUserMediaDevices(const MediaStreamRequest&, MediaDeviceHashSalts&&, Vector<DeviceInfo>& audioDevices, Vector<DeviceInfo>& videoDevices, MediaConstraintType&);
    void validateRequestConstraintsAfterEnumeration(ValidConstraintsHandler&&, InvalidConstraintsHandler&&, const MediaStreamRequest&, MediaDeviceHashSalts&&);
    void enumerateDevices(bool shouldEnumerateCamera, bool shouldEnumerateDisplay, bool shouldEnumerateMicrophone, bool shouldEnumerateSpeakers, CompletionHandler<void()>&&);

    RunLoop::Timer m_debounceTimer;
    void triggerDevicesChangedObservers();

    WeakHashSet<RealtimeMediaSourceCenterObserver> m_observers;

    AudioCaptureFactory* m_audioCaptureFactoryOverride { nullptr };
    VideoCaptureFactory* m_videoCaptureFactoryOverride { nullptr };
    DisplayCaptureFactory* m_displayCaptureFactoryOverride { nullptr };

    bool m_shouldInterruptAudioOnPageVisibilityChange { false };

#if ENABLE(APP_PRIVACY_REPORT)
    OSObjectPtr<tcc_identity_t> m_identity;
#endif

#if ENABLE(EXTENSION_CAPABILITIES)
    String m_currentMediaEnvironment;
#endif

    bool m_useMockCaptureDevices { false };
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)

