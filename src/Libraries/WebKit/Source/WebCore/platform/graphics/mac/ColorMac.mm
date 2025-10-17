/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 7, 2024.
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
#import "ColorMac.h"

#if USE(APPKIT)

#import "GraphicsContextCG.h"
#import "LocalCurrentGraphicsContext.h"
#import <wtf/BlockObjCExceptions.h>
#import <wtf/Lock.h>
#import <wtf/NeverDestroyed.h>
#import <wtf/RetainPtr.h>
#import <wtf/TinyLRUCache.h>

namespace WTF {

template<> RetainPtr<NSColor> TinyLRUCachePolicy<WebCore::Color, RetainPtr<NSColor>>::createValueForKey(const WebCore::Color& color)
{
    return [NSColor colorWithCGColor:cachedCGColor(color).get()];
}

} // namespace WTF

namespace WebCore {

static bool useOldAquaFocusRingColor;

Color oldAquaFocusRingColor()
{
    return SRGBA<uint8_t> { 125, 173, 217 };
}

void setUsesTestModeFocusRingColor(bool newValue)
{
    useOldAquaFocusRingColor = newValue;
}

bool usesTestModeFocusRingColor()
{
    return useOldAquaFocusRingColor;
}

static std::optional<SRGBA<uint8_t>> makeSimpleColorFromNSColor(NSColor *color)
{
    // FIXME: ExtendedColor - needs to handle color spaces.

    if (!color)
        return std::nullopt;

    CGFloat redComponent;
    CGFloat greenComponent;
    CGFloat blueComponent;
    CGFloat alpha;

    BEGIN_BLOCK_OBJC_EXCEPTIONS
    NSColor *rgbColor = [color colorUsingColorSpace:NSColorSpace.deviceRGBColorSpace];
    if (!rgbColor) {
        // The color space conversion above can fail if the NSColor is in the NSPatternColorSpace.
        // These colors are actually a repeating pattern, not just a solid color. To workaround
        // this we simply draw a one pixel image of the color and use that pixel's color.
        // FIXME: It might be better to use an average of the colors in the pattern instead.
        RetainPtr<NSBitmapImageRep> offscreenRep = adoptNS([[NSBitmapImageRep alloc] initWithBitmapDataPlanes:nil pixelsWide:1 pixelsHigh:1
            bitsPerSample:8 samplesPerPixel:4 hasAlpha:YES isPlanar:NO colorSpaceName:NSDeviceRGBColorSpace bytesPerRow:4 bitsPerPixel:32]);
        {
            LocalCurrentCGContext localContext { [NSGraphicsContext graphicsContextWithBitmapImageRep:offscreenRep.get()].CGContext };
            [color drawSwatchInRect:NSMakeRect(0, 0, 1, 1)];
        }
        std::array<NSUInteger, 4> pixel;
        [offscreenRep getPixel:pixel.data() atX:0 y:0];

        return makeFromComponentsClamping<SRGBA<uint8_t>>(pixel[0], pixel[1], pixel[2], pixel[3]);
    }

    [rgbColor getRed:&redComponent green:&greenComponent blue:&blueComponent alpha:&alpha];
    END_BLOCK_OBJC_EXCEPTIONS

    return convertColor<SRGBA<uint8_t>>(SRGBA<float> { static_cast<float>(redComponent), static_cast<float>(greenComponent), static_cast<float>(blueComponent), static_cast<float>(alpha) });
}

Color colorFromCocoaColor(NSColor *color)
{
    return makeSimpleColorFromNSColor(color);
}

Color semanticColorFromNSColor(NSColor *color)
{
    return Color(makeSimpleColorFromNSColor(color), Color::Flags::Semantic);
}

RetainPtr<NSColor> cocoaColor(const Color& color)
{
    if (auto srgb = color.tryGetAsSRGBABytes()) {
        switch (PackedColor::RGBA { *srgb }.value) {
        case PackedColor::RGBA { Color::transparentBlack }.value: {
            static LazyNeverDestroyed<RetainPtr<NSColor>> clearColor;
            static std::once_flag onceFlag;
            std::call_once(onceFlag, [] {
                clearColor.construct([NSColor colorWithSRGBRed:0 green:0 blue:0 alpha:0]);
            });
            return clearColor.get();
        }
        case PackedColor::RGBA { Color::black }.value: {
            static LazyNeverDestroyed<RetainPtr<NSColor>> blackColor;
            static std::once_flag onceFlag;
            std::call_once(onceFlag, [] {
                blackColor.construct([NSColor colorWithSRGBRed:0 green:0 blue:0 alpha:1]);
            });
            return blackColor.get();
        }
        case PackedColor::RGBA { Color::white }.value: {
            static LazyNeverDestroyed<RetainPtr<NSColor>> whiteColor;
            static std::once_flag onceFlag;
            std::call_once(onceFlag, [] {
                whiteColor.construct([NSColor colorWithSRGBRed:1 green:1 blue:1 alpha:1]);
            });
            return whiteColor.get();
        }
        }
    }

    static Lock cachedColorLock;
    Locker locker { cachedColorLock };

    static NeverDestroyed<TinyLRUCache<Color, RetainPtr<NSColor>, 32>> cache;
    return cache.get().get(color);
}

} // namespace WebCore

#endif // USE(APPKIT)
