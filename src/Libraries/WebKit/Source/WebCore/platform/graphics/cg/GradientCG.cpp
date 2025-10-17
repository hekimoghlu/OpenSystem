/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 18, 2023.
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
#include "config.h"
#include "Gradient.h"

#if USE(CG)

#include "GradientRendererCG.h"
#include "GraphicsContextCG.h"
#include <pal/spi/cg/CoreGraphicsSPI.h>

namespace WebCore {

void Gradient::stopsChanged()
{
    m_platformRenderer = { };
}

void Gradient::fill(GraphicsContext& context, const FloatRect& rect)
{
    context.clip(rect);
    paint(context);
}

void Gradient::paint(GraphicsContext& context)
{
    paint(context.platformContext());
}

void Gradient::paint(CGContextRef platformContext)
{
    if (!m_platformRenderer)
        m_platformRenderer = GradientRendererCG { m_colorInterpolationMethod, m_stops.sorted() };

    WTF::switchOn(m_data,
        [&] (const LinearData& data) {
            switch (m_spreadMethod) {
            case GradientSpreadMethod::Repeat:
            case GradientSpreadMethod::Reflect: {
                CGContextStateSaver saveState(platformContext);
                CGGradientDrawingOptions extendOptions = 0;

                FloatPoint gradientVectorNorm(data.point1 - data.point0);
                gradientVectorNorm.normalize();
                CGFloat angle = gradientVectorNorm.isZero() ? 0 : atan2(gradientVectorNorm.y(), gradientVectorNorm.x());
                CGContextRotateCTM(platformContext, angle);

                CGRect boundingBox = CGContextGetClipBoundingBox(platformContext);
                if (CGRectIsInfinite(boundingBox) || CGRectIsEmpty(boundingBox))
                    break;

                CGAffineTransform transform = CGAffineTransformMakeRotation(-angle);
                FloatPoint point0 = CGPointApplyAffineTransform(data.point0, transform);
                FloatPoint point1 = CGPointApplyAffineTransform(data.point1, transform);
                CGFloat dx = point1.x() - point0.x();

                CGFloat pixelSize = CGFAbs(CGContextConvertSizeToUserSpace(platformContext, CGSizeMake(1, 1)).width);
                if (CGFAbs(dx) < pixelSize)
                    dx = dx < 0 ? -pixelSize : pixelSize;

                auto drawLinearGradient = [&](CGFloat start, CGFloat end, bool flip) {
                    CGPoint left = CGPointMake(flip ? end : start, 0);
                    CGPoint right = CGPointMake(flip ? start : end, 0);

                    m_platformRenderer->drawLinearGradient(platformContext, left, right, extendOptions);
                };

                auto isLeftOf = [](CGFloat start, CGFloat end, CGRect boundingBox) -> bool {
                    return std::max(start, end) <= CGRectGetMinX(boundingBox);
                };

                auto isRightOf = [](CGFloat start, CGFloat end, CGRect boundingBox) -> bool {
                    return std::min(start, end) >= CGRectGetMaxX(boundingBox);
                };

                auto isIntersecting = [](CGFloat start, CGFloat end, CGRect boundingBox) -> bool {
                    return std::min(start, end) < CGRectGetMaxX(boundingBox) && CGRectGetMinX(boundingBox) < std::max(start, end);
                };

                bool flip = false;
                CGFloat start = point0.x();

                // Should the points be moved forward towards boundingBox?
                if ((dx > 0 && isLeftOf(start, start + dx, boundingBox)) || (dx < 0 && isRightOf(start, start + dx, boundingBox))) {
                    // Move the 'start' point towards boundingBox.
                    for (; !isIntersecting(start, start + dx, boundingBox); start += dx)
                        flip = !flip && m_spreadMethod == GradientSpreadMethod::Reflect;
                }

                // Draw gradient forward till the points are outside boundingBox.
                for (; isIntersecting(start, start + dx, boundingBox); start += dx) {
                    drawLinearGradient(start, start + dx, flip);
                    flip = !flip && m_spreadMethod == GradientSpreadMethod::Reflect;
                }

                flip = m_spreadMethod == GradientSpreadMethod::Reflect;
                CGFloat end = point0.x();

                // Should the points be moved backward towards boundingBox?
                if ((dx < 0 && isLeftOf(end, end - dx, boundingBox)) || (dx > 0 && isRightOf(end, end - dx, boundingBox))) {
                    // Move the 'end' point towards boundingBox.
                    for (; !isIntersecting(end, end - dx, boundingBox); end -= dx)
                        flip = !flip && m_spreadMethod == GradientSpreadMethod::Reflect;
                }

                // Draw gradient backward till the points are outside boundingBox.
                for (; isIntersecting(end, end - dx, boundingBox); end -= dx) {
                    drawLinearGradient(end - dx, end, flip);
                    flip = !flip && m_spreadMethod == GradientSpreadMethod::Reflect;
                }
                break;
            }
            case GradientSpreadMethod::Pad: {
                CGGradientDrawingOptions extendOptions = kCGGradientDrawsBeforeStartLocation | kCGGradientDrawsAfterEndLocation;
                m_platformRenderer->drawLinearGradient(platformContext, data.point0, data.point1, extendOptions);
                break;
            }
            }
        },
        [&] (const RadialData& data) {
            bool needScaling = data.aspectRatio != 1;
            if (needScaling) {
                CGContextSaveGState(platformContext);
                // Scale from the center of the gradient. We only ever scale non-deprecated gradients,
                // for which point0 == point1.
                ASSERT(data.point0 == data.point1);
                CGContextTranslateCTM(platformContext, data.point0.x(), data.point0.y());
                CGContextScaleCTM(platformContext, 1, 1 / data.aspectRatio);
                CGContextTranslateCTM(platformContext, -data.point0.x(), -data.point0.y());
            }

            CGGradientDrawingOptions extendOptions = kCGGradientDrawsBeforeStartLocation | kCGGradientDrawsAfterEndLocation;
            m_platformRenderer->drawRadialGradient(platformContext, data.point0, data.startRadius, data.point1, data.endRadius, extendOptions);

            if (needScaling)
                CGContextRestoreGState(platformContext);
        },
        [&] (const ConicData& data) {
#if HAVE(CORE_GRAPHICS_CONIC_GRADIENTS)
            CGContextSaveGState(platformContext);
            CGContextTranslateCTM(platformContext, data.point0.x(), data.point0.y());
            CGContextRotateCTM(platformContext, (CGFloat)-M_PI_2);
            CGContextTranslateCTM(platformContext, -data.point0.x(), -data.point0.y());
            m_platformRenderer->drawConicGradient(platformContext, data.point0, data.angleRadians);
            CGContextRestoreGState(platformContext);
#else
            UNUSED_PARAM(data);
#endif
        }
    );
}

}

#endif
