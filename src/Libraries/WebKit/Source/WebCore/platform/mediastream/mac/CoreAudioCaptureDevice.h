/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 8, 2024.
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

#include "CaptureDevice.h"
#include <pal/spi/cf/CoreAudioSPI.h>
#include <wtf/RefPtr.h>
#include <wtf/text/WTFString.h>

typedef struct OpaqueCMClock* CMClockRef;

namespace WebCore {

class CoreAudioCaptureDevice : public CaptureDevice {
public:
    static std::optional<CoreAudioCaptureDevice> create(uint32_t, DeviceType, const String& groupID);
    virtual ~CoreAudioCaptureDevice() = default;

    uint32_t deviceID() const { return m_deviceID; }

    static Vector<AudioDeviceID> relatedAudioDeviceIDs(AudioDeviceID);

private:
    CoreAudioCaptureDevice(uint32_t, const String& persistentID, DeviceType, const String& label, const String& groupID);

    uint32_t m_deviceID { 0 };
};


} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM) && PLATFORM(MAC)

