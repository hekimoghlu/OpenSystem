/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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

#if ENABLE(MEDIA_STREAM) && PLATFORM(IOS_FAMILY)

#include "CaptureDeviceManager.h"
#include <wtf/Lock.h>
#include <wtf/RetainPtr.h>
#include <wtf/WorkQueue.h>

OBJC_CLASS AVAudioSession;
OBJC_CLASS AVAudioSessionPortDescription;
OBJC_CLASS WebAVAudioSessionAvailableInputsListener;

namespace WebCore {

class AVAudioSessionCaptureDevice;
class CaptureDevice;

class AVAudioSessionCaptureDeviceManager final : public CaptureDeviceManager {
    friend class NeverDestroyed<AVAudioSessionCaptureDeviceManager>;
public:
    WEBCORE_EXPORT static AVAudioSessionCaptureDeviceManager& singleton();

    const Vector<CaptureDevice>& captureDevices() final;
    void computeCaptureDevices(CompletionHandler<void()>&&) final;
    const Vector<CaptureDevice>& speakerDevices() const { return m_speakerDevices; }
    std::optional<CaptureDevice> captureDeviceWithPersistentID(CaptureDevice::DeviceType, const String&);

    std::optional<AVAudioSessionCaptureDevice> audioSessionDeviceWithUID(const String&);
    
    void scheduleUpdateCaptureDevices();

    void enableAllDevicesQuery();
    void disableAllDevicesQuery();

    void setPreferredMicrophoneID(const String&);
    const String& preferredMicrophoneID() const { return m_preferredMicrophoneID; }
    void configurePreferredMicrophone();

    WEBCORE_EXPORT void setPreferredSpeakerID(const String&);
    bool isReceiverPreferredSpeaker() const { return m_isReceiverPreferredSpeaker; }

private:
    AVAudioSessionCaptureDeviceManager();
    ~AVAudioSessionCaptureDeviceManager();

    void createAudioSession();
    void refreshAudioCaptureDevices();
    Vector<AVAudioSessionCaptureDevice> retrieveAudioSessionCaptureDevices() const;
    void setAudioCaptureDevices(Vector<AVAudioSessionCaptureDevice>&&);
    bool setPreferredAudioSessionDeviceIDs();
    void notifyNewCurrentMicrophoneDevice(CaptureDevice&&);

    enum class AudioSessionState { NotNeeded, Inactive, Active };

    std::optional<Vector<CaptureDevice>> m_captureDevices;
    Vector<CaptureDevice> m_speakerDevices;
    std::optional<Vector<AVAudioSessionCaptureDevice>> m_audioSessionCaptureDevices;
    RetainPtr<WebAVAudioSessionAvailableInputsListener> m_listener;
    RetainPtr<AVAudioSession> m_audioSession;
    Ref<WorkQueue> m_dispatchQueue;
    String m_preferredMicrophoneID;
    String m_preferredSpeakerID;
    bool m_isReceiverPreferredSpeaker { false };
    bool m_recomputeDevices { true };
    mutable RetainPtr<AVAudioSessionPortDescription> m_lastDefaultMicrophone;
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM) && PLATFORM(IOS_FAMILY)
