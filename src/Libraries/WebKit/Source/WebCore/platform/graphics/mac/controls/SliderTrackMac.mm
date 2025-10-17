/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 18, 2022.
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
#import "SliderTrackMac.h"

#if PLATFORM(MAC)

#import "ColorSpaceCG.h"
#import "FloatRoundedRect.h"
#import "GraphicsContext.h"
#import <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SliderTrackMac);

SliderTrackMac::SliderTrackMac(SliderTrackPart& part, ControlFactoryMac& controlFactory)
    : ControlMac(part, controlFactory)
{
}

FloatRect SliderTrackMac::rectForBounds(const FloatRect& bounds, const ControlStyle& style) const
{
    static constexpr int sliderTrackWidth = 5;
    float trackWidth = sliderTrackWidth * style.zoomFactor;

    auto& sliderTrackPart = owningSliderTrackPart();
    auto rect = bounds;
    
    // Set the height/width and align the location in the center of the difference.
    if (sliderTrackPart.type() == StyleAppearance::SliderHorizontal) {
        rect.setHeight(trackWidth);
        rect.setY(rect.y() + (bounds.height() - rect.height()) / 2);
    } else {
        rect.setWidth(trackWidth);
        rect.setX(rect.x() + (bounds.width() - rect.width()) / 2);
    }

    return rect;
}

static void trackGradientInterpolate(void*, const CGFloat* rawInData, CGFloat* rawOutData)
{
    auto inData = unsafeMakeSpan(rawInData, 1);
    auto outData = unsafeMakeSpan(rawOutData, 4);

    static constexpr std::array dark { 0.0f, 0.0f, 0.0f, 0.678f };
    static constexpr std::array light { 0.0f, 0.0f, 0.0f, 0.13f };
    float a = inData[0];
    for (size_t i = 0; i < 4; ++i)
        outData[i] = (1.0f - a) * dark[i] + a * light[i];
}

void SliderTrackMac::draw(GraphicsContext& context, const FloatRoundedRect& borderRect, float, const ControlStyle& style)
{
    static constexpr int sliderTrackRadius = 2;
    static constexpr IntSize sliderRadius(sliderTrackRadius, sliderTrackRadius);

    CGContextRef cgContext = context.platformContext();
    CGColorSpaceRef cspace = sRGBColorSpaceRef();

    auto& sliderTrackPart = owningSliderTrackPart();

    sliderTrackPart.drawTicks(context, borderRect.rect(), style);

    GraphicsContextStateSaver stateSaver(context);

    auto logicalRect = rectForBounds(borderRect.rect(), style);
    CGContextClipToRect(cgContext, logicalRect);

    struct CGFunctionCallbacks mainCallbacks = { 0, trackGradientInterpolate, NULL };
    RetainPtr<CGFunctionRef> mainFunction = adoptCF(CGFunctionCreate(NULL, 1, NULL, 4, NULL, &mainCallbacks));
    RetainPtr<CGShadingRef> mainShading;
    if (sliderTrackPart.type() == StyleAppearance::SliderVertical)
        mainShading = adoptCF(CGShadingCreateAxial(cspace, CGPointMake(logicalRect.x(),  logicalRect.maxY()), CGPointMake(logicalRect.maxX(), logicalRect.maxY()), mainFunction.get(), false, false));
    else
        mainShading = adoptCF(CGShadingCreateAxial(cspace, CGPointMake(logicalRect.x(),  logicalRect.y()), CGPointMake(logicalRect.x(), logicalRect.maxY()), mainFunction.get(), false, false));

    context.clipRoundedRect(FloatRoundedRect(logicalRect, sliderRadius, sliderRadius, sliderRadius, sliderRadius));
    CGContextDrawShading(cgContext, mainShading.get());
}

} // namespace WebCore

#endif // PLATFORM(MAC)
