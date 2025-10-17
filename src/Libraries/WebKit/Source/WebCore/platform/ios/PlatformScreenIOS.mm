/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 24, 2023.
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
#import "PlatformScreen.h"

#if PLATFORM(IOS_FAMILY)

#import "ContentsFormat.h"
#import "DeprecatedGlobalSettings.h"
#import "FloatRect.h"
#import "FloatSize.h"
#import "GraphicsContextCG.h"
#import "HostWindow.h"
#import "IntRect.h"
#import "LocalFrameView.h"
#import "PlatformCALayerClient.h"
#import "ScreenProperties.h"
#import "WAKWindow.h"
#import "Widget.h"
#import <pal/spi/ios/MobileGestaltSPI.h>
#import <pal/spi/ios/UIKitSPI.h>
#import <pal/system/ios/Device.h>

#import <pal/cocoa/MediaToolboxSoftLink.h>
#import <pal/ios/UIKitSoftLink.h>

namespace WebCore {

int screenDepth(Widget*)
{
    // See <rdar://problem/9378829> for why this is a constant.
    return 24;
}

int screenDepthPerComponent(Widget*)
{
    return screenDepth(nullptr) / 3;
}

bool screenIsMonochrome(Widget*)
{
    return PAL::softLinkUIKitUIAccessibilityIsGrayscaleEnabled();
}

bool screenHasInvertedColors()
{
    if (auto data = screenData(primaryScreenDisplayID()))
        return data->screenHasInvertedColors;

    return PAL::softLinkUIKitUIAccessibilityIsInvertColorsEnabled();
}

ContentsFormat screenContentsFormat(Widget* widget, PlatformCALayerClient* client)
{
#if HAVE(HDR_SUPPORT)
    if (client && client->hdrForImagesEnabled() && screenSupportsHighDynamicRange(widget))
        return ContentsFormat::RGBA16F;
#endif

#if HAVE(IOSURFACE_RGB10)
    if (screenSupportsExtendedColor(widget))
        return ContentsFormat::RGBA10;
#endif

    UNUSED_PARAM(widget);
    UNUSED_PARAM(client);
    return ContentsFormat::RGBA8;
}

bool screenSupportsExtendedColor(Widget*)
{
    if (auto data = screenData(primaryScreenDisplayID()))
        return data->screenSupportsExtendedColor;

    return MGGetBoolAnswer(kMGQHasExtendedColorDisplay);
}

bool screenSupportsHighDynamicRange(Widget*)
{
    if (auto data = screenData(primaryScreenDisplayID()))
        return data->screenSupportsHighDynamicRange;

#if USE(MEDIATOOLBOX)
    if (PAL::isMediaToolboxFrameworkAvailable() && PAL::canLoad_MediaToolbox_MTShouldPlayHDRVideo())
        return PAL::softLink_MediaToolbox_MTShouldPlayHDRVideo(nullptr);
#endif
    return false;
}

DestinationColorSpace screenColorSpace(Widget* widget)
{
#if HAVE(IOSURFACE_RGB10)
    if (screenContentsFormat(widget) == ContentsFormat::RGBA10)
        return DestinationColorSpace { extendedSRGBColorSpaceRef() };
#else
    UNUSED_PARAM(widget);
#endif
    return DestinationColorSpace::SRGB();
}

// These functions scale between screen and page coordinates because JavaScript/DOM operations
// assume that the screen and the page share the same coordinate system.
FloatRect screenRect(Widget* widget)
{
    if (!widget)
        return FloatRect();

    if (NSView *platformWidget = widget->platformWidget()) {
        // WebKit1
        WAKWindow *window = [platformWidget window];
        if (!window)
            return [platformWidget frame];
        CGRect screenRect = { CGPointZero, [window screenSize] };
        return enclosingIntRect(screenRect);
    }
    return enclosingIntRect(FloatRect(FloatPoint(), widget->root()->hostWindow()->overrideScreenSize()));
}

FloatRect screenAvailableRect(Widget* widget)
{
    if (!widget)
        return FloatRect();

    if (NSView *platformWidget = widget->platformWidget()) {
        // WebKit1
        WAKWindow *window = [platformWidget window];
        if (!window)
            return FloatRect();
        CGRect screenRect = { CGPointZero, [window availableScreenSize] };
        return enclosingIntRect(screenRect);
    }
    return enclosingIntRect(FloatRect(FloatPoint(), widget->root()->hostWindow()->overrideAvailableScreenSize()));
}

float screenPPIFactor()
{
    if (auto data = screenData(primaryScreenDisplayID()))
        return data->scaleFactor;

    static float ppiFactor;

    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        int pitch = MGGetSInt32Answer(kMGQMainScreenPitch, 0);
        float scale = MGGetFloat32Answer(kMGQMainScreenScale, 0);

        static const float originalIPhonePPI = 163;
        float mainScreenPPI = (pitch && scale) ? pitch / scale : originalIPhonePPI;
        ppiFactor = mainScreenPPI / originalIPhonePPI;
    });

    return ppiFactor;
}

FloatSize screenSize()
{
    if (PAL::deviceHasIPadCapability() && [[PAL::getUIApplicationClass() sharedApplication] _isClassic])
        return { 320, 480 };

    if (auto data = screenData(primaryScreenDisplayID()))
        return data->screenRect.size();

    return FloatSize([[PAL::getUIScreenClass() mainScreen] _referenceBounds].size);
}

FloatSize availableScreenSize()
{
    if (PAL::deviceHasIPadCapability() && [[PAL::getUIApplicationClass() sharedApplication] _isClassic])
        return { 320, 480 };

    if (auto data = screenData(primaryScreenDisplayID()))
        return data->screenAvailableRect.size();

    return FloatSize([PAL::getUIScreenClass() mainScreen].bounds.size);
}

#if USE(APPLE_INTERNAL_SDK) && __has_include(<WebKitAdditions/PlatformScreenIOS.mm>)
#import <WebKitAdditions/PlatformScreenIOS.mm>
#else
FloatSize overrideScreenSize()
{
    return screenSize();
}
FloatSize overrideAvailableScreenSize()
{
    return availableScreenSize();
}
#endif

float screenScaleFactor(UIScreen *screen)
{
    if (!screen)
        screen = [PAL::getUIScreenClass() mainScreen];

    return screen.scale;
}

ScreenProperties collectScreenProperties()
{
    ScreenProperties screenProperties;

    // FIXME: This displayID doesn't match the synthetic displayIDs we use in iOS WebKit (see WebPageProxy::generateDisplayIDFromPageID()).
    PlatformDisplayID displayID = 0;

    for (UIScreen *screen in [PAL::getUIScreenClass() screens]) {
        ScreenData screenData;

        auto screenAvailableRect = FloatRect { screen.bounds };
        screenAvailableRect.setY(NSMaxY(screen.bounds) - (screenAvailableRect.y() + screenAvailableRect.height())); // flip
        screenData.screenAvailableRect = screenAvailableRect;

        screenData.screenRect = screen._referenceBounds;
        screenData.colorSpace = { screenColorSpace(nullptr) };
        screenData.screenDepth = WebCore::screenDepth(nullptr);
        screenData.screenDepthPerComponent = WebCore::screenDepthPerComponent(nullptr);
        screenData.screenSupportsExtendedColor = WebCore::screenSupportsExtendedColor(nullptr);
        screenData.screenHasInvertedColors = WebCore::screenHasInvertedColors();
        screenData.screenSupportsHighDynamicRange = WebCore::screenSupportsHighDynamicRange(nullptr);
        screenData.scaleFactor = WebCore::screenPPIFactor();

        screenProperties.screenDataMap.set(++displayID, WTFMove(screenData));

        if (screen == [PAL::getUIScreenClass() mainScreen])
            screenProperties.primaryDisplayID = displayID;
    }

    return screenProperties;
}

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY)
