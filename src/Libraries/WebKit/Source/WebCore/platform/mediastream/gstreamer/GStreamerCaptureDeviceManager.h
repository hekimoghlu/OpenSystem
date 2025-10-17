/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 3, 2024.
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

#if ENABLE(MEDIA_STREAM) && USE(GSTREAMER)

#include "DisplayCaptureManager.h"
#include "GRefPtrGStreamer.h"
#include "GStreamerCaptureDevice.h"
#include "GStreamerCapturer.h"
#include "GStreamerVideoCapturer.h"
#include "RealtimeMediaSourceCenter.h"
#include "RealtimeMediaSourceFactory.h"

#include <wtf/Noncopyable.h>

namespace WebCore {

using NodeAndFD = GStreamerVideoCapturer::NodeAndFD;

void teardownGStreamerCaptureDeviceManagers();

class GStreamerCaptureDeviceManager : public CaptureDeviceManager, public RealtimeMediaSourceCenterObserver {
    WTF_MAKE_NONCOPYABLE(GStreamerCaptureDeviceManager)
public:
    GStreamerCaptureDeviceManager();
    ~GStreamerCaptureDeviceManager();
    std::optional<GStreamerCaptureDevice> gstreamerDeviceWithUID(const String&);

    const Vector<CaptureDevice>& captureDevices() final;
    virtual CaptureDevice::DeviceType deviceType() = 0;

    // RealtimeMediaSourceCenterObserver interface.
    void devicesChanged() final;
    void deviceWillBeRemoved(const String& persistentId) final;

    void registerCapturer(RefPtr<GStreamerCapturer>&&);
    void unregisterCapturer(const GStreamerCapturer&);
    void stopCapturing(const String& persistentId);

    void teardown();

private:
    void addDevice(GRefPtr<GstDevice>&&);
    void removeDevice(GRefPtr<GstDevice>&&);
    void stopMonitor();
    void refreshCaptureDevices();

    GRefPtr<GstDeviceMonitor> m_deviceMonitor;
    Vector<GStreamerCaptureDevice> m_gstreamerDevices;
    Vector<CaptureDevice> m_devices;
    Vector<RefPtr<GStreamerCapturer>> m_capturers;
    bool m_isTearingDown { false };
};

class GStreamerAudioCaptureDeviceManager final : public GStreamerCaptureDeviceManager {
    friend class NeverDestroyed<GStreamerAudioCaptureDeviceManager>;
public:
    static GStreamerAudioCaptureDeviceManager& singleton();
    CaptureDevice::DeviceType deviceType() final { return CaptureDevice::DeviceType::Microphone; }

    // ref() and deref() do nothing because this object is a singleton.
    void ref() const final { }
    void deref() const final { }
};

class GStreamerVideoCaptureDeviceManager final : public GStreamerCaptureDeviceManager {
    friend class NeverDestroyed<GStreamerVideoCaptureDeviceManager>;
public:
    static GStreamerVideoCaptureDeviceManager& singleton();
    CaptureDevice::DeviceType deviceType() final { return CaptureDevice::DeviceType::Camera; }

    // ref() and deref() do nothing because this object is a singleton.
    void ref() const final { }
    void deref() const final { }
};

class GStreamerDisplayCaptureDeviceManager final : public DisplayCaptureManager {
    friend class NeverDestroyed<GStreamerDisplayCaptureDeviceManager>;
public:
    static GStreamerDisplayCaptureDeviceManager& singleton();
    const Vector<CaptureDevice>& captureDevices() final { return m_devices; };
    void computeCaptureDevices(CompletionHandler<void()>&&) final;
    CaptureSourceOrError createDisplayCaptureSource(const CaptureDevice&, MediaDeviceHashSalts&&, const MediaConstraints*);

    enum PipeWireOutputType {
        Monitor = 1 << 0,
        Window = 1 << 1
    };

    void stopSource(const String& persistentID);

    // DisplayCaptureManager interface
    bool requiresCaptureDevicesEnumeration() const final { return true; }

protected:
    void notifyResponse(GVariant* parameters) { m_currentResponseCallback(parameters); }

private:
    GStreamerDisplayCaptureDeviceManager();
    ~GStreamerDisplayCaptureDeviceManager();

    using ResponseCallback = CompletionHandler<void(GVariant*)>;

    void waitResponseSignal(const char* objectPath, ResponseCallback&& = [](GVariant*) { });

    Vector<CaptureDevice> m_devices;

    struct Session {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;
        WTF_MAKE_NONCOPYABLE(Session);
        Session(const NodeAndFD& nodeAndFd, String&& path)
            : nodeAndFd(nodeAndFd)
            , path(WTFMove(path)) { }

        ~Session()
        {
            close(nodeAndFd.second);
        }

        NodeAndFD nodeAndFd;
        String path;
    };
    HashMap<String, std::unique_ptr<Session>> m_sessions;

    GRefPtr<GDBusProxy> m_proxy;
    ResponseCallback m_currentResponseCallback;
};
}

#endif // ENABLE(MEDIA_STREAM) && USE(GSTREAMER)
