/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 6, 2022.
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
#import "ColorCocoa.h"

#if PLATFORM(IOS_FAMILY)

#import "ColorSpaceCG.h"
#import <UIKit/UIKit.h>

namespace WebCore {

Color colorFromCocoaColor(UIColor *color)
{
    if (!color)
        return { };

    // FIXME: ExtendedColor - needs to handle color spaces.

    // FIXME: Make this work for a UIColor that was created from a pattern or a DispayP3 color.
    CGFloat redComponent;
    CGFloat greenComponent;
    CGFloat blueComponent;
    CGFloat alpha;

    BOOL success = [color getRed:&redComponent green:&greenComponent blue:&blueComponent alpha:&alpha];
    if (!success) {
        // The color space conversion above can fail if the UIColor is in an incompatible color space.
        // To workaround this we simply draw a one pixel image of the color and use that pixel's color.
        uint8_t pixel[4];
        auto bitmapContext = adoptCF(CGBitmapContextCreate(pixel, 1, 1, 8, 4, sRGBColorSpaceRef(), kCGImageAlphaPremultipliedLast));

        CGContextSetFillColorWithColor(bitmapContext.get(), color.CGColor);
        CGContextFillRect(bitmapContext.get(), CGRectMake(0, 0, 1, 1));

        return makeFromComponentsClamping<SRGBA<uint8_t>>(pixel[0], pixel[1], pixel[2], pixel[3]);
    }

    return convertColor<SRGBA<uint8_t>>(SRGBA<float> { static_cast<float>(redComponent), static_cast<float>(greenComponent), static_cast<float>(blueComponent), static_cast<float>(alpha) });
}

} // namespace WebCore

#endif
