/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 29, 2025.
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

#include "CaptureDevice.h"
#include <wtf/RetainPtr.h>

OBJC_CLASS AVAudioSessionPortDescription;

namespace WebCore {

class AVAudioSessionCaptureDevice : public CaptureDevice {
public:
    static AVAudioSessionCaptureDevice createInput(AVAudioSessionPortDescription *deviceInput, AVAudioSessionPortDescription *defaultInput);
    static AVAudioSessionCaptureDevice createOutput(AVAudioSessionPortDescription *deviceOutput, AVAudioSessionPortDescription *defaultOutput);
    virtual ~AVAudioSessionCaptureDevice() = default;

    AVAudioSessionCaptureDevice isolatedCopy() &&;

    bool isBuiltin() const;
    bool isLineInOrOut() const;
    bool isHeadset() const;

private:
    AVAudioSessionCaptureDevice(AVAudioSessionPortDescription *deviceInput, AVAudioSessionPortDescription *defaultInput, CaptureDevice::DeviceType);
    AVAudioSessionCaptureDevice(const String& persistentId, DeviceType, const String& label, const String& groupId, bool isEnabled, bool isDefault, bool isMock, RetainPtr<AVAudioSessionPortDescription>&&);

    RetainPtr<AVAudioSessionPortDescription> m_description;
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM) && PLATFORM(IOS_FAMILY)
