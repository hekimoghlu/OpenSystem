/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 17, 2023.
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

#if ENABLE(WIRELESS_PLAYBACK_TARGET) && !PLATFORM(IOS_FAMILY)

#include "AVPlaybackTargetPicker.h"
#include <wtf/TZoneMalloc.h>

OBJC_CLASS AVOutputDeviceMenuController;
OBJC_CLASS WebAVOutputDeviceMenuControllerHelper;

namespace WebCore {
class AVOutputDeviceMenuControllerTargetPicker;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::AVOutputDeviceMenuControllerTargetPicker> : std::true_type { };
}

namespace WebCore {

class AVOutputDeviceMenuControllerTargetPicker final : public AVPlaybackTargetPicker {
    WTF_MAKE_TZONE_ALLOCATED(AVOutputDeviceMenuControllerTargetPicker);
    WTF_MAKE_NONCOPYABLE(AVOutputDeviceMenuControllerTargetPicker);
public:
    explicit AVOutputDeviceMenuControllerTargetPicker(AVPlaybackTargetPickerClient&);
    virtual ~AVOutputDeviceMenuControllerTargetPicker();

    void availableDevicesDidChange();
    void currentDeviceDidChange();

private:
    void showPlaybackTargetPicker(NSView *, const FloatRect&, bool checkActiveRoute, bool useDarkAppearance) final;
    void startingMonitoringPlaybackTargets() final;
    void stopMonitoringPlaybackTargets() final;
    void invalidatePlaybackTargets() final;
    bool externalOutputDeviceAvailable() final;
    AVOutputContext* outputContext() final;

    AVOutputDeviceMenuController *devicePicker();

    RetainPtr<AVOutputDeviceMenuController> m_outputDeviceMenuController;
    RetainPtr<WebAVOutputDeviceMenuControllerHelper> m_outputDeviceMenuControllerDelegate;
    bool m_showingMenu { false };
};

} // namespace WebCore

#endif // ENABLE(WIRELESS_PLAYBACK_TARGET)
