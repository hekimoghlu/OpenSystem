/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 19, 2021.
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
#ifndef MediaPlaybackTargetPickerMock_h
#define MediaPlaybackTargetPickerMock_h

#if ENABLE(WIRELESS_PLAYBACK_TARGET) && !PLATFORM(IOS_FAMILY)

#include "MediaPlaybackTargetContext.h"
#include "MediaPlaybackTargetPicker.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class MediaPlaybackTargetPickerMock;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::MediaPlaybackTargetPickerMock> : std::true_type { };
}

namespace WebCore {

class MediaPlaybackTargetPickerMock final : public MediaPlaybackTargetPicker, public CanMakeWeakPtr<MediaPlaybackTargetPickerMock> {
    WTF_MAKE_TZONE_ALLOCATED(MediaPlaybackTargetPickerMock);
    WTF_MAKE_NONCOPYABLE(MediaPlaybackTargetPickerMock);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(MediaPlaybackTargetPickerMock);
public:
    explicit MediaPlaybackTargetPickerMock(MediaPlaybackTargetPicker::Client&);

    virtual ~MediaPlaybackTargetPickerMock();

    void showPlaybackTargetPicker(PlatformView*, const FloatRect&, bool checkActiveRoute, bool useDarkAppearance) override;
    void startingMonitoringPlaybackTargets() override;
    void stopMonitoringPlaybackTargets() override;
    void invalidatePlaybackTargets() override;

    void setState(const String&, MediaPlaybackTargetContext::MockState);
    void dismissPopup();

private:
    bool externalOutputDeviceAvailable() override;
    Ref<MediaPlaybackTarget> playbackTarget() override;

    String m_deviceName;
    MediaPlaybackTargetContext::MockState m_state { MediaPlaybackTargetContext::MockState::Unknown };
    bool m_showingMenu { false };
};

} // namespace WebCore

#endif // ENABLE(WIRELESS_PLAYBACK_TARGET) && !PLATFORM(IOS_FAMILY)

#endif // WebContextMenuProxyMac_h
