/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 25, 2023.
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

#include "MediaDeviceInfo.h"
#include "MediaTrackCapabilities.h"
#include "RealtimeMediaSourceCapabilities.h"

namespace WebCore {

struct CaptureDeviceWithCapabilities;

class InputDeviceInfo final : public MediaDeviceInfo {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(InputDeviceInfo);
public:
    static Ref<InputDeviceInfo> create(CaptureDeviceWithCapabilities&& device, String&& saltedDeviceId, String&& saltedGroupId) { return adoptRef(*new InputDeviceInfo(WTFMove(device), WTFMove(saltedDeviceId), WTFMove(saltedGroupId))); }

    MediaTrackCapabilities getCapabilities() const;

private:
    InputDeviceInfo(CaptureDeviceWithCapabilities&&, String&& saltedDeviceId, String&& saltedGroupId);

    RealtimeMediaSourceCapabilities m_capabilities;
};

}

#endif
