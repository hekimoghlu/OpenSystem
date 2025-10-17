/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 12, 2021.
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

#if ENABLE(MEDIA_STREAM) && PLATFORM(MAC)

#include "CaptureDeviceManager.h"
#include <CoreAudio/CoreAudio.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class CoreAudioCaptureDevice;

class CoreAudioCaptureDeviceManager final : public CaptureDeviceManager {
    friend class NeverDestroyed<CoreAudioCaptureDeviceManager>;
public:
    WEBCORE_EXPORT static CoreAudioCaptureDeviceManager& singleton();

    const Vector<CaptureDevice>& captureDevices() final;
    std::optional<CaptureDevice> captureDeviceWithPersistentID(CaptureDevice::DeviceType, const String&);

    std::optional<CoreAudioCaptureDevice> coreAudioDeviceWithUID(const String&);
    const Vector<CaptureDevice>& speakerDevices() const { return m_speakerDevices; }

    void setFilterTapEnabledDevices(bool doFiltering) { m_filterTapEnabledDevices = doFiltering; }

private:
    CoreAudioCaptureDeviceManager() = default;
    virtual ~CoreAudioCaptureDeviceManager();

    Vector<CoreAudioCaptureDevice>& coreAudioCaptureDevices();

    enum class NotifyIfDevicesHaveChanged { Notify, DoNotNotify };
    void refreshAudioCaptureDevices(NotifyIfDevicesHaveChanged);
    void scheduleUpdateCaptureDevices();

    Vector<CaptureDevice> m_captureDevices;
    Vector<CaptureDevice> m_speakerDevices;
    Vector<CoreAudioCaptureDevice> m_coreAudioCaptureDevices;
    bool m_wasRefreshAudioCaptureDevicesScheduled { false };
    bool m_filterTapEnabledDevices { false };
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM) && PLATFORM(MAC)
