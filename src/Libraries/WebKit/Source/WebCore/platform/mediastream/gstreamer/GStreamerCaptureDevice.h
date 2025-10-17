/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 23, 2022.
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

#include "CaptureDevice.h"
#include "GStreamerCommon.h"

namespace WebCore {

class GStreamerCaptureDevice : public CaptureDevice {
public:
    GStreamerCaptureDevice(GRefPtr<GstDevice>&& device, const String& persistentId, DeviceType type, const String& label, const String& groupId = emptyString())
        : CaptureDevice(persistentId, type, label, groupId)
        , m_device(WTFMove(device))
    {
    }

    WARN_UNUSED_RETURN GRefPtr<GstCaps> caps() const { return adoptGRef(gst_device_get_caps(m_device.get())); }
    GstDevice* device() { return m_device.get(); }

private:
    GRefPtr<GstDevice> m_device;
};

}

#endif // ENABLE(MEDIA_STREAM)  && USE(GSTREAMER)
