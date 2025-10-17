/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 27, 2022.
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
#import "AVOutputDeviceMenuControllerTargetPicker.h"

#if ENABLE(WIRELESS_PLAYBACK_TARGET) && !PLATFORM(IOS_FAMILY)

#import "FloatRect.h"
#import "Logging.h"
#import <objc/runtime.h>
#import <pal/spi/cocoa/AVFoundationSPI.h>
#import <pal/spi/cocoa/AVKitSPI.h>
#import <wtf/MainThread.h>
#import <wtf/TZoneMallocInlines.h>

#import <pal/cf/CoreMediaSoftLink.h>
#import <pal/cocoa/AVFoundationSoftLink.h>

SOFTLINK_AVKIT_FRAMEWORK()
SOFT_LINK_CLASS_OPTIONAL(AVKit, AVOutputDeviceMenuController)

using namespace WebCore;

static NSString *externalOutputDeviceAvailableKeyName = @"externalOutputDeviceAvailable";
static NSString *externalOutputDevicePickedKeyName = @"externalOutputDevicePicked";

@interface WebAVOutputDeviceMenuControllerHelper : NSObject {
    WeakPtr<AVOutputDeviceMenuControllerTargetPicker> m_callback;
}

- (instancetype)initWithCallback:(WeakPtr<AVOutputDeviceMenuControllerTargetPicker>&&)callback;
- (void)clearCallback;
- (void)observeValueForKeyPath:(id)keyPath ofObject:(id)object change:(NSDictionary *)change context:(void *)context;
@end

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(AVOutputDeviceMenuControllerTargetPicker);

AVOutputDeviceMenuControllerTargetPicker::AVOutputDeviceMenuControllerTargetPicker(AVPlaybackTargetPickerClient& client)
    : AVPlaybackTargetPicker(client)
    , m_outputDeviceMenuControllerDelegate(adoptNS([[WebAVOutputDeviceMenuControllerHelper alloc] initWithCallback:*this]))
{
}

AVOutputDeviceMenuControllerTargetPicker::~AVOutputDeviceMenuControllerTargetPicker()
{
    [m_outputDeviceMenuControllerDelegate clearCallback];
}

AVOutputDeviceMenuController *AVOutputDeviceMenuControllerTargetPicker::devicePicker()
{
    if (!getAVOutputDeviceMenuControllerClass())
        return nullptr;

    if (!m_outputDeviceMenuController) {
        RetainPtr<AVOutputContext> context = adoptNS([PAL::allocAVOutputContextInstance() init]);
        m_outputDeviceMenuController = adoptNS([allocAVOutputDeviceMenuControllerInstance() initWithOutputContext:context.get()]);

        [m_outputDeviceMenuController addObserver:m_outputDeviceMenuControllerDelegate.get() forKeyPath:externalOutputDeviceAvailableKeyName options:NSKeyValueObservingOptionNew context:nullptr];
        [m_outputDeviceMenuController addObserver:m_outputDeviceMenuControllerDelegate.get() forKeyPath:externalOutputDevicePickedKeyName options:NSKeyValueObservingOptionNew context:nullptr];

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
        if (m_outputDeviceMenuController.get().externalOutputDeviceAvailable)
            availableDevicesDidChange();
ALLOW_DEPRECATED_DECLARATIONS_END
    }

    return m_outputDeviceMenuController.get();
}

void AVOutputDeviceMenuControllerTargetPicker::availableDevicesDidChange()
{
    if (client())
        client()->availableDevicesChanged();
}

void AVOutputDeviceMenuControllerTargetPicker::currentDeviceDidChange()
{
    if (client())
        client()->currentDeviceChanged();
}

void AVOutputDeviceMenuControllerTargetPicker::showPlaybackTargetPicker(NSView *, const FloatRect& location, bool hasActiveRoute, bool useDarkAppearance)
{
    if (!client() || m_showingMenu)
        return;

    m_showingMenu = true;

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    bool targetSelected = [devicePicker() showMenuForRect:location appearanceName:(useDarkAppearance ? NSAppearanceNameVibrantDark : NSAppearanceNameVibrantLight) allowReselectionOfSelectedOutputDevice:!hasActiveRoute];
ALLOW_DEPRECATED_DECLARATIONS_END

    if (!client())
        return;

    if (targetSelected != hasActiveRoute)
        currentDeviceDidChange();
    else if (!targetSelected && !hasActiveRoute)
        client()->pickerWasDismissed();

    m_showingMenu = false;
}

void AVOutputDeviceMenuControllerTargetPicker::startingMonitoringPlaybackTargets()
{
    devicePicker();
}

void AVOutputDeviceMenuControllerTargetPicker::stopMonitoringPlaybackTargets()
{
    // Nothing to do, AirPlay takes care of this automatically.
}

void AVOutputDeviceMenuControllerTargetPicker::invalidatePlaybackTargets()
{
    if (m_outputDeviceMenuController) {
        [m_outputDeviceMenuController removeObserver:m_outputDeviceMenuControllerDelegate.get() forKeyPath:externalOutputDeviceAvailableKeyName];
        [m_outputDeviceMenuController removeObserver:m_outputDeviceMenuControllerDelegate.get() forKeyPath:externalOutputDevicePickedKeyName];
        m_outputDeviceMenuController = nullptr;
    }
    currentDeviceDidChange();
}

bool AVOutputDeviceMenuControllerTargetPicker::externalOutputDeviceAvailable()
{
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    return devicePicker().externalOutputDeviceAvailable;
ALLOW_DEPRECATED_DECLARATIONS_END
}

AVOutputContext * AVOutputDeviceMenuControllerTargetPicker::outputContext()
{
    return m_outputDeviceMenuController ? [m_outputDeviceMenuController outputContext] : nullptr;
}

} // namespace WebCore

@implementation WebAVOutputDeviceMenuControllerHelper
- (instancetype)initWithCallback:(WeakPtr<AVOutputDeviceMenuControllerTargetPicker>&&)callback
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

- (void)observeValueForKeyPath:(id)keyPath ofObject:(id)object change:(NSDictionary *)change context:(void *)context
{
    UNUSED_PARAM(object);
    UNUSED_PARAM(change);
    UNUSED_PARAM(context);

    if (!m_callback)
        return;

    if (![keyPath isEqualToString:externalOutputDeviceAvailableKeyName] && ![keyPath isEqualToString:externalOutputDevicePickedKeyName])
        return;

    callOnMainThread([self, protectedSelf = retainPtr(self), keyPath = retainPtr(keyPath)] {
        if (!m_callback)
            return;

        if ([keyPath isEqualToString:externalOutputDeviceAvailableKeyName])
            m_callback->availableDevicesDidChange();
        else if ([keyPath isEqualToString:externalOutputDevicePickedKeyName])
            m_callback->currentDeviceDidChange();
    });
}
@end

#endif // ENABLE(WIRELESS_PLAYBACK_TARGET)
