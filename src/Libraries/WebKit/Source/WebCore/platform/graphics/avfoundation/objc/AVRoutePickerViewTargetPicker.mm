/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 8, 2022.
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
#import "AVRoutePickerViewTargetPicker.h"

#if ENABLE(WIRELESS_PLAYBACK_TARGET) && HAVE(AVROUTEPICKERVIEW)

#import "FloatRect.h"
#import "Logging.h"
#import <AVFoundation/AVRouteDetector.h>
#import <pal/spi/cocoa/AVFoundationSPI.h>
#import <pal/spi/cocoa/AVKitSPI.h>
#import <wtf/MainThread.h>
#include <wtf/TZoneMallocInlines.h>

#import <pal/cf/CoreMediaSoftLink.h>
#import <pal/cocoa/AVFoundationSoftLink.h>

SOFTLINK_AVKIT_FRAMEWORK()
SOFT_LINK_CLASS_OPTIONAL(AVKit, AVRoutePickerView)

using namespace WebCore;

@interface WebAVRoutePickerViewHelper : NSObject <AVRoutePickerViewDelegate> {
    WeakPtr<AVRoutePickerViewTargetPicker> m_callback;
}

- (instancetype)initWithCallback:(WeakPtr<AVRoutePickerViewTargetPicker>&&)callback;
- (void)clearCallback;
- (void)notificationHandler:(NSNotification *)notification;
- (void)routePickerViewDidEndPresentingRoutes:(AVRoutePickerView *)routePickerView;
@end

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(AVRoutePickerViewTargetPicker);

bool AVRoutePickerViewTargetPicker::isAvailable()
{
    static bool available;
    static std::once_flag flag;
    std::call_once(flag, [] () {
        if (!getAVRoutePickerViewClass())
            return;

        if (auto picker = adoptNS([allocAVRoutePickerViewInstance() init]))
            available = [picker respondsToSelector:@selector(showRoutePickingControlsForOutputContext:relativeToRect:ofView:)];
    });

    return available;
}

AVRoutePickerViewTargetPicker::AVRoutePickerViewTargetPicker(AVPlaybackTargetPickerClient& client)
    : AVPlaybackTargetPicker(client)
    , m_routePickerViewDelegate(adoptNS([[WebAVRoutePickerViewHelper alloc] initWithCallback:*this]))
{
    ASSERT(isAvailable());
}

AVRoutePickerViewTargetPicker::~AVRoutePickerViewTargetPicker()
{
    [m_routePickerViewDelegate clearCallback];
}

AVOutputContext * AVRoutePickerViewTargetPicker::outputContextInternal()
{
    if (!m_outputContext) {
        m_outputContext = [PAL::getAVOutputContextClass() iTunesAudioContext];
        ASSERT(m_outputContext);
        if (m_outputContext)
            [[NSNotificationCenter defaultCenter] addObserver:m_routePickerViewDelegate.get() selector:@selector(notificationHandler:) name:PAL::AVOutputContextOutputDevicesDidChangeNotification object:m_outputContext.get()];
    }

    return m_outputContext.get();
}

AVRoutePickerView *AVRoutePickerViewTargetPicker::devicePicker()
{
    if (!m_routePickerView) {
        m_routePickerView = adoptNS([allocAVRoutePickerViewInstance() init]);
        [m_routePickerView setDelegate:m_routePickerViewDelegate.get()];
    }

    return m_routePickerView.get();
}

AVRouteDetector *AVRoutePickerViewTargetPicker::routeDetector()
{
    if (!m_routeDetector) {
        m_routeDetector = adoptNS([PAL::allocAVRouteDetectorInstance() init]);
        [[NSNotificationCenter defaultCenter] addObserver:m_routePickerViewDelegate.get() selector:@selector(notificationHandler:) name:PAL::AVRouteDetectorMultipleRoutesDetectedDidChangeNotification object:m_routeDetector.get()];
        if ([m_routeDetector multipleRoutesDetected])
            availableDevicesDidChange();
    }

    return m_routeDetector.get();
}

void AVRoutePickerViewTargetPicker::showPlaybackTargetPicker(NSView *view, const FloatRect& rectInScreenCoordinates, bool hasActiveRoute, bool useDarkAppearance)
{
    if (!client())
        return;

    auto *picker = devicePicker();
    if (useDarkAppearance)
        picker.routeListAlwaysHasDarkAppearance = YES;

    m_hadActiveRoute = hasActiveRoute;

    auto rectInWindowCoordinates = [view.window convertRectFromScreen:NSMakeRect(rectInScreenCoordinates.x(), rectInScreenCoordinates.y(), 1.0, 1.0)];
    auto rectInViewCoordinates = [view convertRect:rectInWindowCoordinates fromView:view];
    [picker showRoutePickingControlsForOutputContext:outputContextInternal() relativeToRect:rectInViewCoordinates ofView:view];
}

void AVRoutePickerViewTargetPicker::startingMonitoringPlaybackTargets()
{
    m_ignoreNextMultipleRoutesDetectedDidChangeNotification = false;

    routeDetector().routeDetectionEnabled = YES;
}

void AVRoutePickerViewTargetPicker::stopMonitoringPlaybackTargets()
{
    if (!m_routeDetector)
        return;

    // `-[AVRouteDetector multipleRoutesDetected]` will always return `NO` if route detection is
    // disabled and `-[AVRouteDetector setRouteDetectionEnabled:]` will always dispatch a
    // `AVRouteDetectorMultipleRoutesDetectedDidChange` notification, so ignore the next one in
    // order to prevent the cached value in the WebProcess from always being `false` when the last
    // JS `"webkitplaybacktargetavailabilitychanged"` event listener is removed.
    m_ignoreNextMultipleRoutesDetectedDidChangeNotification = true;

    [m_routeDetector setRouteDetectionEnabled:NO];
}

bool AVRoutePickerViewTargetPicker::externalOutputDeviceAvailable()
{
    return routeDetector().multipleRoutesDetected;
}

AVOutputContext * AVRoutePickerViewTargetPicker::outputContext()
{
    return m_outputContext.get();
}

void AVRoutePickerViewTargetPicker::invalidatePlaybackTargets()
{
    if (m_routeDetector) {
        [[NSNotificationCenter defaultCenter] removeObserver:m_routePickerViewDelegate.get() name:PAL::AVRouteDetectorMultipleRoutesDetectedDidChangeNotification object:m_routeDetector.get()];
        [m_routeDetector setRouteDetectionEnabled:NO];
        m_routePickerView = nullptr;
    }

    if (m_outputContext) {
        [[NSNotificationCenter defaultCenter] removeObserver:m_routePickerViewDelegate.get() name:PAL::AVOutputContextOutputDevicesDidChangeNotification object:m_outputContext.get()];
        m_outputContext = nullptr;
    }

    if (m_routePickerView) {
        [m_routePickerView setDelegate:nil];
        m_routePickerView = nullptr;
    }
    currentDeviceDidChange();
}
void AVRoutePickerViewTargetPicker::availableDevicesDidChange()
{
    if (m_ignoreNextMultipleRoutesDetectedDidChangeNotification) {
        m_ignoreNextMultipleRoutesDetectedDidChangeNotification = false;
        return;
    }

    if (client())
        client()->availableDevicesChanged();
}

bool AVRoutePickerViewTargetPicker::hasActiveRoute() const
{
    if (!m_outputContext)
        return false;

    if ([m_outputContext respondsToSelector:@selector(supportsMultipleOutputDevices)] && [m_outputContext respondsToSelector:@selector(outputDevices)]&& [m_outputContext supportsMultipleOutputDevices]) {
        for (AVOutputDevice *outputDevice in [m_outputContext outputDevices]) {
            if (outputDevice.deviceFeatures & (AVOutputDeviceFeatureVideo | AVOutputDeviceFeatureAudio))
                return true;
        }

        return false;
    }

    if ([m_outputContext respondsToSelector:@selector(outputDevice)]) {
        if (auto *outputDevice = [m_outputContext outputDevice])
            return outputDevice.deviceFeatures & (AVOutputDeviceFeatureVideo | AVOutputDeviceFeatureAudio);
    }

    return [m_outputContext deviceName];
}

void AVRoutePickerViewTargetPicker::currentDeviceDidChange()
{
    auto haveActiveRoute = hasActiveRoute();
    if (!client() || m_hadActiveRoute == haveActiveRoute)
        return;

    m_hadActiveRoute = haveActiveRoute;
    client()->currentDeviceChanged();
}

void AVRoutePickerViewTargetPicker::devicePickerWasDismissed()
{
    if (!client())
        return;
    
    client()->pickerWasDismissed();
    currentDeviceDidChange();
}

} // namespace WebCore

@implementation WebAVRoutePickerViewHelper
- (instancetype)initWithCallback:(WeakPtr<AVRoutePickerViewTargetPicker>&&)callback
{
    if (!(self = [super init]))
        return nil;

    m_callback = WTFMove(callback);

    return self;
}

- (void)clearCallback
{
    m_callback = nil;
}

- (void)routePickerViewDidEndPresentingRoutes:(AVRoutePickerView *)routePickerView
{
    UNUSED_PARAM(routePickerView);

    if (!m_callback)
        return;

    callOnMainThread([self, protectedSelf = retainPtr(self)] {
        if (!m_callback)
            return;

        m_callback->devicePickerWasDismissed();
    });
}

- (void)notificationHandler:(NSNotification *)notification
{
    UNUSED_PARAM(notification);

    if (!m_callback)
        return;

    callOnMainThread([self, protectedSelf = retainPtr(self), notification = retainPtr(notification)] {
        if (!m_callback)
            return;

        if ([[notification name] isEqualToString:PAL::AVOutputContextOutputDevicesDidChangeNotification])
            m_callback->currentDeviceDidChange();
        else if ([[notification name] isEqualToString:PAL::AVRouteDetectorMultipleRoutesDetectedDidChangeNotification])
            m_callback->availableDevicesDidChange();
    });
}

@end

#endif // ENABLE(WIRELESS_PLAYBACK_TARGET)
