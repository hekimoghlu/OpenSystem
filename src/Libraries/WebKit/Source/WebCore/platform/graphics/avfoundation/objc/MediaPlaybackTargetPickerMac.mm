/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 26, 2024.
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
#import "config.h"
#import "MediaPlaybackTargetPickerMac.h"

#if ENABLE(WIRELESS_PLAYBACK_TARGET) && !PLATFORM(IOS_FAMILY)

#import "AVOutputDeviceMenuControllerTargetPicker.h"
#import "FloatRect.h"
#import "Logging.h"
#import "MediaPlaybackTargetCocoa.h"
#import <objc/runtime.h>
#import <pal/spi/cocoa/AVFoundationSPI.h>
#import <pal/spi/cocoa/AVKitSPI.h>
#import <wtf/MainThread.h>
#include <wtf/TZoneMallocInlines.h>

#if HAVE(AVROUTEPICKERVIEW)
#import "AVRoutePickerViewTargetPicker.h"
#endif

#import <pal/cf/CoreMediaSoftLink.h>
#import <pal/cocoa/AVFoundationSoftLink.h>

SOFTLINK_AVKIT_FRAMEWORK()
SOFT_LINK_CLASS_OPTIONAL(AVKit, AVOutputDeviceMenuController)

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MediaPlaybackTargetPickerMac);

MediaPlaybackTargetPickerMac::MediaPlaybackTargetPickerMac(MediaPlaybackTargetPicker::Client& client)
    : MediaPlaybackTargetPicker(client)
{
}

MediaPlaybackTargetPickerMac::~MediaPlaybackTargetPickerMac()
{
    setClient(nullptr);
}

bool MediaPlaybackTargetPickerMac::externalOutputDeviceAvailable()
{
    return routePicker().externalOutputDeviceAvailable();
}

Ref<MediaPlaybackTarget> MediaPlaybackTargetPickerMac::playbackTarget()
{
    return WebCore::MediaPlaybackTargetCocoa::create(MediaPlaybackTargetContextCocoa(routePicker().outputContext()));
}

AVPlaybackTargetPicker& MediaPlaybackTargetPickerMac::routePicker()
{
    if (m_routePicker)
        return *m_routePicker;

#if HAVE(AVROUTEPICKERVIEW)
    if (AVRoutePickerViewTargetPicker::isAvailable())
        m_routePicker = makeUnique<AVRoutePickerViewTargetPicker>(*this);
    else
#endif
        m_routePicker = makeUnique<AVOutputDeviceMenuControllerTargetPicker>(*this);
    
    return *m_routePicker;
}

void MediaPlaybackTargetPickerMac::showPlaybackTargetPicker(PlatformView* view, const FloatRect& location, bool hasActiveRoute, bool useDarkAppearance)
{
    routePicker().showPlaybackTargetPicker(view, location, hasActiveRoute, useDarkAppearance);
}

void MediaPlaybackTargetPickerMac::startingMonitoringPlaybackTargets()
{
    LOG(Media, "MediaPlaybackTargetPickerMac::startingMonitoringPlaybackTargets");

    routePicker().startingMonitoringPlaybackTargets();
}

void MediaPlaybackTargetPickerMac::stopMonitoringPlaybackTargets()
{
    LOG(Media, "MediaPlaybackTargetPickerMac::stopMonitoringPlaybackTargets");
    routePicker().stopMonitoringPlaybackTargets();
}

void MediaPlaybackTargetPickerMac::invalidatePlaybackTargets()
{
    LOG(Media, "MediaPlaybackTargetPickerMac::invalidatePlaybackTargets");

    m_routePicker = nullptr;
    currentDeviceDidChange();
}

void MediaPlaybackTargetPickerMac::pickerWasDismissed()
{
    playbackTargetPickerWasDismissed();
}

void MediaPlaybackTargetPickerMac::availableDevicesChanged()
{
    availableDevicesDidChange();
}

void MediaPlaybackTargetPickerMac::currentDeviceChanged()
{
    currentDeviceDidChange();
}


} // namespace WebCore


#endif // ENABLE(WIRELESS_PLAYBACK_TARGET)
