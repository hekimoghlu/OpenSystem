/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 23, 2023.
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

#include "CaptureDevice.h"
#include "ContextDestructionObserver.h"
#include "ScriptWrappable.h"
#include <wtf/text/WTFString.h>

namespace WebCore {

class MediaDeviceInfo : public RefCounted<MediaDeviceInfo>, public ScriptWrappable {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MediaDeviceInfo);
public:
    enum class Kind { Audioinput, Audiooutput, Videoinput };

    static Ref<MediaDeviceInfo> create(const String&, const String&, const String&, Kind);
    virtual ~MediaDeviceInfo() = default;

    const String& label() const { return m_label; }
    const String& deviceId() const { return m_deviceId; }
    const String& groupId() const { return m_groupId; }
    Kind kind() const { return m_kind; }

protected:
    MediaDeviceInfo(const String&, const String&, const String&, Kind);

private:
    const String m_label;
    const String m_deviceId;
    const String m_groupId;
    const Kind m_kind;
};

MediaDeviceInfo::Kind toMediaDeviceInfoKind(CaptureDevice::DeviceType);

typedef Vector<RefPtr<MediaDeviceInfo>> MediaDeviceInfoVector;

}

#endif
