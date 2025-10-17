/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 20, 2022.
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

#if PLATFORM(COCOA) && ENABLE(MEDIA_STREAM)

#include "Connection.h"
#include "MessageReceiver.h"
#include "RemoteVideoFrameObjectHeap.h"
#include "UserMediaCaptureManager.h"
#include <WebCore/CaptureDevice.h>
#include <WebCore/IntDegrees.h>
#include <WebCore/OrientationNotifier.h>
#include <WebCore/ProcessIdentity.h>
#include <WebCore/RealtimeMediaSource.h>
#include <WebCore/RealtimeMediaSourceIdentifier.h>
#include <pal/spi/cocoa/TCCSPI.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>

namespace WebKit {
class UserMediaCaptureManagerProxy;
}

namespace WebCore {
class PlatformMediaSessionManager;
class SharedMemory;
struct VideoPresetData;
}

namespace WebKit {

class WebProcessProxy;
class UserMediaCaptureManagerProxySourceProxy;

class UserMediaCaptureManagerProxy : public IPC::MessageReceiver, public RefCounted<UserMediaCaptureManagerProxy> {
    WTF_MAKE_TZONE_ALLOCATED(UserMediaCaptureManagerProxy);
public:
    class ConnectionProxy {
    public:
        virtual ~ConnectionProxy() = default;
        virtual void addMessageReceiver(IPC::ReceiverName, IPC::MessageReceiver&) = 0;
        virtual void removeMessageReceiver(IPC::ReceiverName) = 0;
        virtual IPC::Connection& connection() = 0;
        Ref<IPC::Connection> protectedConnection() { return connection(); }
        virtual bool willStartCapture(WebCore::CaptureDevice::DeviceType, WebCore::PageIdentifier) const = 0;
        virtual Logger& logger() = 0;
        Ref<Logger> protectedLogger() { return logger(); };
        virtual bool setCaptureAttributionString() { return true; }
        virtual const WebCore::ProcessIdentity& resourceOwner() const = 0;
#if ENABLE(APP_PRIVACY_REPORT)
        virtual void setTCCIdentity() { }
#endif
#if ENABLE(EXTENSION_CAPABILITIES)
        virtual bool setCurrentMediaEnvironment(WebCore::PageIdentifier) { return false; };
#endif
        virtual void startProducingData(WebCore::CaptureDevice::DeviceType) { }
        virtual RemoteVideoFrameObjectHeap* remoteVideoFrameObjectHeap() { return nullptr; }

        virtual void startMonitoringCaptureDeviceRotation(WebCore::PageIdentifier, const String&) { }
        virtual void stopMonitoringCaptureDeviceRotation(WebCore::PageIdentifier, const String&) { }
        virtual std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const = 0;
    };
    static Ref<UserMediaCaptureManagerProxy> create(UniqueRef<ConnectionProxy>&&);
    ~UserMediaCaptureManagerProxy();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    void close();
    void clear();

    void setOrientation(WebCore::IntDegrees);
    void rotationAngleForCaptureDeviceChanged(const String&, WebCore::VideoFrameRotation);

    void didReceiveMessageFromGPUProcess(IPC::Connection& connection, IPC::Decoder& decoder) { didReceiveMessage(connection, decoder); }
    bool didReceiveSyncMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&);

    bool hasSourceProxies() const;

    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;

private:
    explicit UserMediaCaptureManagerProxy(UniqueRef<ConnectionProxy>&&);

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    using CreateSourceCallback = CompletionHandler<void(const WebCore::CaptureSourceError&, const WebCore::RealtimeMediaSourceSettings&, const WebCore::RealtimeMediaSourceCapabilities&)>;
    void createMediaSourceForCaptureDeviceWithConstraints(WebCore::RealtimeMediaSourceIdentifier, const WebCore::CaptureDevice& deviceID, WebCore::MediaDeviceHashSalts&&, WebCore::MediaConstraints&&, bool shouldUseGPUProcessRemoteFrames, WebCore::PageIdentifier, CreateSourceCallback&&);
    void startProducingData(WebCore::RealtimeMediaSourceIdentifier, WebCore::PageIdentifier);
    void stopProducingData(WebCore::RealtimeMediaSourceIdentifier);
    void removeSource(WebCore::RealtimeMediaSourceIdentifier);
    void capabilities(WebCore::RealtimeMediaSourceIdentifier, CompletionHandler<void(WebCore::RealtimeMediaSourceCapabilities&&)>&&);
    void applyConstraints(WebCore::RealtimeMediaSourceIdentifier, WebCore::MediaConstraints&&);
    void clone(WebCore::RealtimeMediaSourceIdentifier clonedID, WebCore::RealtimeMediaSourceIdentifier cloneID, WebCore::PageIdentifier);
    void endProducingData(WebCore::RealtimeMediaSourceIdentifier);
    void setShouldApplyRotation(WebCore::RealtimeMediaSourceIdentifier, bool shouldApplyRotation);
    void setIsInBackground(WebCore::RealtimeMediaSourceIdentifier, bool);
    void isPowerEfficient(WebCore::RealtimeMediaSourceIdentifier, CompletionHandler<void(bool)>&&);

    using TakePhotoCallback = CompletionHandler<void(Expected<std::pair<Vector<uint8_t>, String>, String>&&)>;
    void takePhoto(WebCore::RealtimeMediaSourceIdentifier, WebCore::PhotoSettings&&, TakePhotoCallback&&);

    using GetPhotoCapabilitiesCallback = CompletionHandler<void(Expected<WebCore::PhotoCapabilities, String>&&)>;
    void getPhotoCapabilities(WebCore::RealtimeMediaSourceIdentifier, GetPhotoCapabilitiesCallback&&);

    using GetPhotoSettingsCallback = CompletionHandler<void(Expected<WebCore::PhotoSettings, String>&&)>;
    void getPhotoSettings(WebCore::RealtimeMediaSourceIdentifier, GetPhotoSettingsCallback&&);

    WebCore::CaptureSourceOrError createMicrophoneSource(const WebCore::CaptureDevice&, WebCore::MediaDeviceHashSalts&&, const WebCore::MediaConstraints*, WebCore::PageIdentifier);
    WebCore::CaptureSourceOrError createCameraSource(const WebCore::CaptureDevice&, WebCore::MediaDeviceHashSalts&&, WebCore::PageIdentifier);

    using SerialAction = Function<Ref<GenericPromise>()>;
    void queueAndProcessSerialAction(SerialAction&&);

    friend class UserMediaCaptureManagerProxySourceProxy;
    HashMap<WebCore::RealtimeMediaSourceIdentifier, Ref<UserMediaCaptureManagerProxySourceProxy>> m_proxies;
    UniqueRef<ConnectionProxy> m_connectionProxy;
    WebCore::OrientationNotifier m_orientationNotifier { 0 };
    Ref<GenericPromise> m_pendingAction { GenericPromise::createAndResolve() };

    struct PageSources {
        ThreadSafeWeakPtr<WebCore::RealtimeMediaSource> microphoneSource;
        ThreadSafeWeakHashSet<WebCore::RealtimeMediaSource> cameraSources;
    };
    HashMap<WebCore::PageIdentifier, PageSources> m_pageSources;
};

}

#endif
