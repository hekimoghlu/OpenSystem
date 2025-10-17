/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 9, 2024.
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
#ifndef MediaPlaybackTargetPickerMac_h
#define MediaPlaybackTargetPickerMac_h

#if ENABLE(WIRELESS_PLAYBACK_TARGET) && !PLATFORM(IOS_FAMILY)

#include "AVPlaybackTargetPicker.h"
#include "MediaPlaybackTargetPicker.h"
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class MediaPlaybackTargetPickerMac final : public MediaPlaybackTargetPicker, public AVPlaybackTargetPickerClient {
    WTF_MAKE_TZONE_ALLOCATED(MediaPlaybackTargetPickerMac);
    WTF_MAKE_NONCOPYABLE(MediaPlaybackTargetPickerMac);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(MediaPlaybackTargetPickerMac);
public:
    explicit MediaPlaybackTargetPickerMac(MediaPlaybackTargetPicker::Client&);
    virtual ~MediaPlaybackTargetPickerMac();

    void showPlaybackTargetPicker(PlatformView*, const FloatRect&, bool checkActiveRoute, bool useDarkAppearance) final;
    void startingMonitoringPlaybackTargets() final;
    void stopMonitoringPlaybackTargets() final;
    void invalidatePlaybackTargets() final;

private:
    bool externalOutputDeviceAvailable() final;
    Ref<MediaPlaybackTarget> playbackTarget() final;

    // AVPlaybackTargetPickerClient
    void pickerWasDismissed() final;
    void availableDevicesChanged() final;
    void currentDeviceChanged() final;

    AVPlaybackTargetPicker& routePicker();

    std::unique_ptr<AVPlaybackTargetPicker> m_routePicker;
};

} // namespace WebCore

#endif // ENABLE(WIRELESS_PLAYBACK_TARGET)

#endif // WebContextMenuProxyMac_h
