/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 15, 2023.
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
#import "GraphicsContext.h"

#import "DisplayListRecorder.h"
#import "Font.h"
#import "GraphicsContextCG.h"
#import "IOSurface.h"
#import "IntRect.h"
#import <CoreText/CoreText.h>
#import <pal/spi/cg/CoreGraphicsSPI.h>
#import <pal/spi/cocoa/FeatureFlagsSPI.h>
#import <pal/spi/mac/NSGraphicsSPI.h>
#import <wtf/SoftLinking.h>
#import <wtf/StdLibExtras.h>

#if ENABLE(MULTI_REPRESENTATION_HEIC)
#import "MultiRepresentationHEICMetrics.h"
#endif

#if USE(APPKIT)
#import <AppKit/AppKit.h>
#endif

#if PLATFORM(IOS_FAMILY)
#import "Color.h"
#import "WKGraphics.h"
#import <UIKit/UIKit.h>
#import <pal/ios/UIKitSoftLink.h>
#import <pal/spi/ios/UIKitSPI.h>
#endif

@class NSColor;

// FIXME: More of this should use CoreGraphics instead of AppKit.
// FIXME: More of this should move into GraphicsContextCG.cpp.

namespace WebCore {

// NSColor, NSBezierPath, and NSGraphicsContext calls do not raise exceptions
// so we don't block exceptions.

#if ENABLE(MULTI_REPRESENTATION_HEIC)

ImageDrawResult GraphicsContext::drawMultiRepresentationHEIC(Image& image, const Font& font, const FloatRect& destination, ImagePaintingOptions options)
{
    RetainPtr multiRepresentationHEIC = image.adapter().multiRepresentationHEIC();
    if (!multiRepresentationHEIC)
        return ImageDrawResult::DidNothing;

    RefPtr imageBuffer = createScaledImageBuffer(destination.size(), scaleFactor(), DestinationColorSpace::SRGB(), RenderingMode::Unaccelerated, RenderingMethod::Local);
    if (!imageBuffer)
        return ImageDrawResult::DidNothing;

    CGContextRef cgContext = imageBuffer->context().platformContext();

    CGContextScaleCTM(cgContext, 1, -1);
    CGContextTranslateCTM(cgContext, 0, -destination.height());

    // FIXME (rdar://123044459): This needs to account for vertical writing modes.
    CGContextSetTextPosition(cgContext, 0, font.metricsForMultiRepresentationHEIC().descent);

    CTFontDrawImageFromAdaptiveImageProviderAtPoint(font.getCTFont(), multiRepresentationHEIC.get(), CGContextGetTextPosition(cgContext), cgContext);

    auto orientation = options.orientation();
    if (orientation == ImageOrientation::Orientation::FromImage)
        orientation = image.orientation();

    drawImageBuffer(*imageBuffer, destination, { options, orientation });

    return ImageDrawResult::DidDraw;
}

#endif

void GraphicsContextCG::drawFocusRing(const Path& path, float, const Color& color)
{
    if (path.isEmpty())
        return;

    CGFocusRingStyle focusRingStyle;
#if USE(APPKIT)
    NSInitializeCGFocusRingStyleForTime(NSFocusRingOnly, &focusRingStyle, std::numeric_limits<double>::max());
#else
    focusRingStyle.version = 0;
    focusRingStyle.tint = kCGFocusRingTintBlue;
    focusRingStyle.ordering = kCGFocusRingOrderingNone;
    focusRingStyle.alpha = [PAL::getUIFocusRingStyleClass() maxAlpha];
    focusRingStyle.radius = [PAL::getUIFocusRingStyleClass() borderThickness];
    focusRingStyle.threshold = [PAL::getUIFocusRingStyleClass() alphaThreshold];
    focusRingStyle.bounds = CGRectZero;
#endif

    // We want to respect the CGContext clipping and also not overpaint any
    // existing focus ring. The way to do this is set accumulate to
    // -1. According to CoreGraphics, the reasoning for this behavior has been
    // lost in time.
    focusRingStyle.accumulate = -1;
    auto style = adoptCF(CGStyleCreateFocusRingWithColor(&focusRingStyle, cachedCGColor(color).get()));

    CGContextRef platformContext = this->platformContext();

    CGContextStateSaver stateSaver(platformContext);

    CGContextSetStyle(platformContext, style.get());
    CGContextBeginPath(platformContext);
    CGContextAddPath(platformContext, path.platformPath());

    CGContextFillPath(platformContext);
}

void GraphicsContextCG::drawFocusRing(const Vector<FloatRect>& rects, float outlineOffset, float outlineWidth, const Color& color)
{
    Path path;
    for (const auto& rect : rects) {
        auto r = rect;
        r.inflate(-outlineOffset);
        path.addRect(r);
    }
    drawFocusRing(path, outlineWidth, color);
}

static inline void setPatternPhaseInUserSpace(CGContextRef context, CGPoint phasePoint)
{
    CGAffineTransform userToBase = getUserToBaseCTM(context);
    CGPoint phase = CGPointApplyAffineTransform(phasePoint, userToBase);

    CGContextSetPatternPhase(context, CGSizeMake(phase.x, phase.y));
}

static inline void drawDotsForDocumentMarker(CGContextRef context, const FloatRect& rect, DocumentMarkerLineStyle style)
{
    // We want to find the number of full dots, so we're solving the equations:
    // dotDiameter = height
    // dotDiameter / dotGap = 13.247 / 9.457
    // numberOfGaps = numberOfDots - 1
    // dotDiameter * numberOfDots + dotGap * numberOfGaps = width

    auto width = rect.width();
    auto dotDiameter = rect.height();
    auto dotGap = dotDiameter * 9.457 / 13.247;
    auto numberOfDots = (width + dotGap) / (dotDiameter + dotGap);
    auto numberOfWholeDots = static_cast<unsigned>(numberOfDots);
    auto numberOfWholeGaps = numberOfWholeDots - 1;

    // Center the dots
    auto offset = (width - (dotDiameter * numberOfWholeDots + dotGap * numberOfWholeGaps)) / 2;

    CGContextStateSaver stateSaver { context };
    CGContextSetFillColorWithColor(context, cachedCGColor(style.color).get());
    for (unsigned i = 0; i < numberOfWholeDots; ++i) {
        auto location = rect.location();
        location.move(offset + i * (dotDiameter + dotGap), 0);
        auto size = FloatSize(dotDiameter, dotDiameter);
        CGContextAddEllipseInRect(context, FloatRect(location, size));
    }
    CGContextSetCompositeOperation(context, kCGCompositeSover);
    CGContextFillPath(context);
}

#if HAVE(AUTOCORRECTION_ENHANCEMENTS)

static inline void drawRoundedRectForDocumentMarker(CGContextRef context, const FloatRect& rect, DocumentMarkerLineStyle style)
{
    CGContextStateSaver stateSaver { context };
    CGContextSetFillColorWithColor(context, cachedCGColor(style.color).get());
    CGContextSetCompositeOperation(context, kCGCompositeSover);

    auto radius = rect.height() / 2.0;
    auto minX = rect.x();
    auto maxX = rect.maxX();
    auto minY = rect.y();
    auto maxY = rect.maxY();
    auto midY = (minY + maxY) / 2.0;

    CGContextMoveToPoint(context, minX + radius, maxY);
    CGContextAddArc(context, minX + radius, midY, radius, piOverTwoDouble, 3 * piOverTwoDouble, 0);
    CGContextAddLineToPoint(context, maxX - radius, minY);
    CGContextAddArc(context, maxX - radius, midY, radius, 3 * piOverTwoDouble, piOverTwoDouble, 0);
    CGContextClosePath(context);
    CGContextFillPath(context);
}

#endif

void GraphicsContextCG::drawDotsForDocumentMarker(const FloatRect& rect, DocumentMarkerLineStyle style)
{
#if HAVE(AUTOCORRECTION_ENHANCEMENTS)
    if (style.mode == DocumentMarkerLineStyleMode::AutocorrectionReplacement) {
        drawRoundedRectForDocumentMarker(this->platformContext(), rect, style);
        return;
    }
#endif
    WebCore::drawDotsForDocumentMarker(this->platformContext(), rect, style);
}

} // namespace WebCore
