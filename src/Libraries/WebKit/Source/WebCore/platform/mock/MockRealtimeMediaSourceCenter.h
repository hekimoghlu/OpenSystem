/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 1, 2025.
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

#include "DisplayCaptureManager.h"
#include "MockMediaDevice.h"
#include "MockRealtimeAudioSource.h"
#include "MockRealtimeVideoSource.h"
#include "RealtimeMediaSourceCenter.h"

namespace WebCore {

class MockRealtimeMediaSourceCenter {
public:
    WEBCORE_EXPORT static MockRealtimeMediaSourceCenter& singleton();

    WEBCORE_EXPORT static void setMockRealtimeMediaSourceCenterEnabled(bool);
    WEBCORE_EXPORT static bool mockRealtimeMediaSourceCenterEnabled();

    WEBCORE_EXPORT static void setDevices(Vector<MockMediaDevice>&&);
    WEBCORE_EXPORT static void addDevice(const MockMediaDevice&);
    WEBCORE_EXPORT static void removeDevice(const String&);
    WEBCORE_EXPORT static void setDeviceIsEphemeral(const String&, bool);
    WEBCORE_EXPORT static void resetDevices();
    WEBCORE_EXPORT static void setMockCaptureDevicesInterrupted(bool isCameraInterrupted, bool isMicrophoneInterrupted);

    WEBCORE_EXPORT void triggerMockCaptureConfigurationChange(bool forMicrophone, bool forDisplay);

    void setMockAudioCaptureEnabled(bool isEnabled) { m_isMockAudioCaptureEnabled = isEnabled; }
    void setMockVideoCaptureEnabled(bool isEnabled) { m_isMockVideoCaptureEnabled = isEnabled; }
    void setMockDisplayCaptureEnabled(bool isEnabled) { m_isMockDisplayCaptureEnabled = isEnabled; }

    static Vector<CaptureDevice>& microphoneDevices();
    static Vector<CaptureDevice>& speakerDevices();
    static Vector<CaptureDevice>& videoDevices();
    WEBCORE_EXPORT static Vector<CaptureDevice>& displayDevices();

    static std::optional<MockMediaDevice> mockDeviceWithPersistentID(const String&);
    static std::optional<CaptureDevice> captureDeviceWithPersistentID(CaptureDevice::DeviceType, const String&);

    CaptureDeviceManager& audioCaptureDeviceManager() { return m_audioCaptureDeviceManager; }
    CaptureDeviceManager& videoCaptureDeviceManager() { return m_videoCaptureDeviceManager; }
    DisplayCaptureManager& displayCaptureDeviceManager() { return m_displayCaptureDeviceManager; }

private:
    MockRealtimeMediaSourceCenter() = default;
    friend NeverDestroyed<MockRealtimeMediaSourceCenter>;

    AudioCaptureFactory& audioCaptureFactory();
    VideoCaptureFactory& videoCaptureFactory();
    DisplayCaptureFactory& displayCaptureFactory();

    class MockAudioCaptureDeviceManager final : public CaptureDeviceManager {
    private:
        const Vector<CaptureDevice>& captureDevices() final { return MockRealtimeMediaSourceCenter::microphoneDevices(); }
        std::optional<CaptureDevice> captureDeviceWithPersistentID(CaptureDevice::DeviceType type, const String& id) final { return MockRealtimeMediaSourceCenter::captureDeviceWithPersistentID(type, id); }
    };
    class MockVideoCaptureDeviceManager final : public CaptureDeviceManager {
    private:
        const Vector<CaptureDevice>& captureDevices() final { return MockRealtimeMediaSourceCenter::videoDevices(); }
        std::optional<CaptureDevice> captureDeviceWithPersistentID(CaptureDevice::DeviceType type, const String& id) final { return MockRealtimeMediaSourceCenter::captureDeviceWithPersistentID(type, id); }
    };
    class MockDisplayCaptureDeviceManager final : public DisplayCaptureManager {
    private:
        const Vector<CaptureDevice>& captureDevices() final { return MockRealtimeMediaSourceCenter::displayDevices(); }
        std::optional<CaptureDevice> captureDeviceWithPersistentID(CaptureDevice::DeviceType type, const String& id) final { return MockRealtimeMediaSourceCenter::captureDeviceWithPersistentID(type, id); }
    };

    MockAudioCaptureDeviceManager m_audioCaptureDeviceManager;
    MockVideoCaptureDeviceManager m_videoCaptureDeviceManager;
    MockDisplayCaptureDeviceManager m_displayCaptureDeviceManager;

    bool m_isMockAudioCaptureEnabled { true };
    bool m_isMockVideoCaptureEnabled { true };
    bool m_isMockDisplayCaptureEnabled { true };
    bool m_isEnabled { false };
};

}

#endif // MockRealtimeMediaSourceCenter_h
