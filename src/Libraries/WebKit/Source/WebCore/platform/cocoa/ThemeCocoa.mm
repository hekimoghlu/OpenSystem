/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 24, 2024.
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
#import "ThemeCocoa.h"

#import "ApplePayLogoSystemImage.h"
#import "FontCascade.h"
#import "GeometryUtilities.h"
#import "GraphicsContext.h"
#import "ImageBuffer.h"
#import <dlfcn.h>

namespace WebCore {

void ThemeCocoa::drawNamedImage(const String& name, GraphicsContext& context, const FloatSize& size) const
{
    if (name == "wireless-playback"_s) {
        GraphicsContextStateSaver stateSaver(context);
        context.setFillColor(Color::black);

        FloatSize wirelessPlaybackSrcSize(32, 24.016);
        auto largestRect = largestRectWithAspectRatioInsideRect(wirelessPlaybackSrcSize.aspectRatio(), FloatRect(FloatPoint::zero(), size));
        context.translate(largestRect.x(), largestRect.y());
        context.scale(largestRect.width() / wirelessPlaybackSrcSize.width());

        Path outline;
        outline.moveTo(FloatPoint(24.066, 18));
        outline.addLineTo(FloatPoint(22.111, 16));
        outline.addLineTo(FloatPoint(30, 16));
        outline.addLineTo(FloatPoint(30, 2));
        outline.addLineTo(FloatPoint(2, 2));
        outline.addLineTo(FloatPoint(2, 16));
        outline.addLineTo(FloatPoint(9.908, 16));
        outline.addLineTo(FloatPoint(7.953, 18));
        outline.addLineTo(FloatPoint(0, 18));
        outline.addLineTo(FloatPoint(0, 0));
        outline.addLineTo(FloatPoint(32, 0));
        outline.addLineTo(FloatPoint(32, 18));
        outline.addLineTo(FloatPoint(24.066, 18));
        outline.closeSubpath();
        outline.moveTo(FloatPoint(26.917, 24.016));
        outline.addLineTo(FloatPoint(5.040, 24.016));
        outline.addLineTo(FloatPoint(15.978, 12.828));
        outline.addLineTo(FloatPoint(26.917, 24.016));
        outline.closeSubpath();

        context.fillPath(outline);
        return;
    }

#if ENABLE(APPLE_PAY)
    if (name == "apple-pay-logo-black"_s) {
        context.drawSystemImage(ApplePayLogoSystemImage::create(ApplePayLogoStyle::Black), FloatRect(FloatPoint::zero(), size));
        return;
    }

    if (name == "apple-pay-logo-white"_s) {
        context.drawSystemImage(ApplePayLogoSystemImage::create(ApplePayLogoStyle::White), FloatRect(FloatPoint::zero(), size));
        return;
    }
#endif

    Theme::drawNamedImage(name, context, size);
}

}
