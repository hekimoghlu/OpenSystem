/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#import "DragImage.h"

#if PLATFORM(IOS_FAMILY)

#import "Document.h"
#import "Element.h"
#import "FloatRoundedRect.h"
#import "FontCascade.h"
#import "FontPlatformData.h"
#import "GeometryUtilities.h"
#import "GraphicsContext.h"
#import "GraphicsContextCG.h"
#import "Image.h"
#import "LocalFrame.h"
#import "NotImplemented.h"
#import "Page.h"
#import "Range.h"
#import "SimpleRange.h"
#import "StringTruncator.h"
#import "TextIndicator.h"
#import "TextRun.h"
#import <CoreGraphics/CoreGraphics.h>
#import <CoreText/CoreText.h>
#import <UIKit/UIColor.h>
#import <UIKit/UIFont.h>
#import <UIKit/UIGraphicsImageRenderer.h>
#import <UIKit/UIImage.h>
#import <pal/ios/UIKitSoftLink.h>
#import <wtf/NeverDestroyed.h>

namespace WebCore {

#if ENABLE(DRAG_SUPPORT)

IntSize dragImageSize(DragImageRef image)
{
    return IntSize(CGImageGetWidth(image.get()), CGImageGetHeight(image.get()));
}

DragImageRef scaleDragImage(DragImageRef image, FloatSize scale)
{
    CGSize imageSize = CGSizeMake(scale.width() * CGImageGetWidth(image.get()), scale.height() * CGImageGetHeight(image.get()));
    CGRect imageRect = { CGPointZero, imageSize };

    RetainPtr<UIGraphicsImageRenderer> render = adoptNS([PAL::allocUIGraphicsImageRendererInstance() initWithSize:imageSize]);
    UIImage *imageCopy = [render imageWithActions:^(UIGraphicsImageRendererContext *rendererContext) {
        CGContextRef context = rendererContext.CGContext;
        CGContextTranslateCTM(context, 0, imageSize.height);
        CGContextScaleCTM(context, 1, -1);
        CGContextDrawImage(context, imageRect, image.get());
    }];
    return imageCopy.CGImage;
}

static float maximumAllowedDragImageArea = 600 * 1024;

DragImageRef createDragImageFromImage(Image* image, ImageOrientation orientation)
{
    if (!image || !image->width() || !image->height())
        return nil;

    float adjustedImageScale = 1;
    CGSize imageSize(image->size());
    if (imageSize.width * imageSize.height > maximumAllowedDragImageArea) {
        auto adjustedSize = roundedIntSize(sizeWithAreaAndAspectRatio(maximumAllowedDragImageArea, imageSize.width / imageSize.height));
        adjustedImageScale = adjustedSize.width() / imageSize.width;
        imageSize = adjustedSize;
    }

    RetainPtr<UIGraphicsImageRenderer> render = adoptNS([PAL::allocUIGraphicsImageRendererInstance() initWithSize:imageSize]);
    UIImage *imageCopy = [render imageWithActions:^(UIGraphicsImageRendererContext *rendererContext) {
        GraphicsContextCG context(rendererContext.CGContext);
        context.translate(0, imageSize.height);
        context.scale({ adjustedImageScale, -adjustedImageScale });
        context.drawImage(*image, FloatPoint(), { orientation });
    }];
    return imageCopy.CGImage;
}

void deleteDragImage(DragImageRef)
{
}

static FontCascade cascadeForSystemFont(CGFloat size)
{
    UIFont *font = [PAL::getUIFontClass() systemFontOfSize:size];
    return FontCascade(FontPlatformData(adoptCF(CTFontCreateWithName((CFStringRef)font.fontName, font.pointSize, nil)), font.pointSize));
}

DragImageRef createDragImageForLink(Element& linkElement, URL& url, const String& title, TextIndicatorData& indicatorData, float)
{
    // FIXME: Most of this can go away once we can use UIURLDragPreviewView unconditionally.
    constexpr CGFloat dragImagePadding = 10;
    static const NeverDestroyed titleFontCascade = cascadeForSystemFont(16);
    static const NeverDestroyed urlFontCascade = cascadeForSystemFont(14);

    auto topString(title.trim(deprecatedIsSpaceOrNewline));
    auto bottomString([(NSURL *)url absoluteString]);
    if (topString.isEmpty()) {
        topString = bottomString;
        bottomString = emptyString();
    }

    static CGFloat maxTextWidth = 320;
    auto truncatedTopString = StringTruncator::rightTruncate(topString, maxTextWidth, titleFontCascade);
    auto truncatedBottomString = StringTruncator::centerTruncate(bottomString, maxTextWidth, urlFontCascade);
    CGFloat textWidth = std::max(StringTruncator::width(truncatedTopString, titleFontCascade), StringTruncator::width(truncatedBottomString, urlFontCascade));
    CGFloat textHeight = truncatedBottomString.isEmpty() ? 22 : 44;

    CGRect imageRect = CGRectMake(0, 0, textWidth + 2 * dragImagePadding, textHeight + 2 * dragImagePadding);

    auto renderer = adoptNS([PAL::allocUIGraphicsImageRendererInstance() initWithSize:imageRect.size]);
    auto image = [renderer imageWithActions:^(UIGraphicsImageRendererContext *rendererContext) {
        GraphicsContextCG context(rendererContext.CGContext);
        context.translate(0, CGRectGetHeight(imageRect));
        context.scale({ 1, -1 });
        context.fillRoundedRect(FloatRoundedRect(imageRect, FloatRoundedRect::Radii(4)), Color::white);
        titleFontCascade.get().drawText(context, TextRun(truncatedTopString), FloatPoint(dragImagePadding, 18 + dragImagePadding));
        if (!truncatedBottomString.isEmpty())
            urlFontCascade.get().drawText(context, TextRun(truncatedBottomString), FloatPoint(dragImagePadding, 40 + dragImagePadding));
    }];

    constexpr OptionSet<TextIndicatorOption> defaultLinkIndicatorOptions {
        TextIndicatorOption::TightlyFitContent,
        TextIndicatorOption::RespectTextColor,
        TextIndicatorOption::UseBoundingRectAndPaintAllContentForComplexRanges,
        TextIndicatorOption::ExpandClipBeyondVisibleRect,
        TextIndicatorOption::ComputeEstimatedBackgroundColor
    };

    if (auto textIndicator = TextIndicator::createWithRange(makeRangeSelectingNodeContents(linkElement), defaultLinkIndicatorOptions, TextIndicatorPresentationTransition::None, FloatSize()))
        indicatorData = textIndicator->data();

    return image.CGImage;
}

DragImageRef createDragImageIconForCachedImageFilename(const String&)
{
    notImplemented();
    return nullptr;
}

DragImageRef platformAdjustDragImageForDeviceScaleFactor(DragImageRef image, float)
{
    // On iOS, we just create the drag image at the right device scale factor, so we don't need to scale it by 1 / deviceScaleFactor later.
    return image;
}

constexpr OptionSet<TextIndicatorOption> defaultSelectionDragImageTextIndicatorOptions {
    TextIndicatorOption::ExpandClipBeyondVisibleRect,
    TextIndicatorOption::PaintAllContent,
    TextIndicatorOption::UseSelectionRectForSizing,
    TextIndicatorOption::ComputeEstimatedBackgroundColor
};

DragImageRef createDragImageForSelection(LocalFrame& frame, TextIndicatorData& indicatorData, bool forceBlackText)
{
    if (auto document = frame.document())
        document->updateLayout();

    auto options = defaultSelectionDragImageTextIndicatorOptions;
    if (!forceBlackText)
        options.add(TextIndicatorOption::RespectTextColor);

    auto textIndicator = TextIndicator::createWithSelectionInFrame(frame, options, TextIndicatorPresentationTransition::None, FloatSize());
    if (!textIndicator)
        return nullptr;

    auto image = textIndicator->contentImage();
    if (image)
        indicatorData = textIndicator->data();
    else
        return nullptr;

    FloatRect imageRect(0, 0, image->width(), image->height());
    if (auto page = frame.page())
        imageRect.scale(1 / page->deviceScaleFactor());


    auto renderer = adoptNS([PAL::allocUIGraphicsImageRendererInstance() initWithSize:imageRect.size()]);
    return [renderer imageWithActions:^(UIGraphicsImageRendererContext *rendererContext) {
        GraphicsContextCG context(rendererContext.CGContext);
        // FIXME: The context flip here should not be necessary, and suggests that somewhere else in the regular
        // drag initiation flow, we unnecessarily flip the graphics context.
        context.translate(0, imageRect.height());
        context.scale({ 1, -1 });
        context.drawImage(*image, imageRect);
    }].CGImage;
}

DragImageRef dissolveDragImageToFraction(DragImageRef image, float)
{
    notImplemented();
    return image;
}

DragImageRef createDragImageForRange(LocalFrame& frame, const SimpleRange& range, bool forceBlackText)
{
    if (auto document = frame.document())
        document->updateLayout();

    if (range.collapsed())
        return nil;

    auto options = defaultSelectionDragImageTextIndicatorOptions;
    if (!forceBlackText)
        options.add(TextIndicatorOption::RespectTextColor);

    auto textIndicator = TextIndicator::createWithRange(range, options, TextIndicatorPresentationTransition::None);
    if (!textIndicator || !textIndicator->contentImage())
        return nil;

    auto& image = *textIndicator->contentImage();
    auto render = adoptNS([PAL::allocUIGraphicsImageRendererInstance() initWithSize:image.size()]);
    UIImage *finalImage = [render imageWithActions:[&image](UIGraphicsImageRendererContext *rendererContext) {
        GraphicsContextCG context(rendererContext.CGContext);
        context.drawImage(image, FloatPoint());
    }];

    return finalImage.CGImage;
}

DragImageRef createDragImageForColor(const Color& color, const FloatRect& elementRect, float pageScaleFactor, Path& visiblePath)
{
    FloatRect imageRect { 0, 0, elementRect.width() * pageScaleFactor, elementRect.height() * pageScaleFactor };
    FloatRoundedRect swatch { imageRect, FloatRoundedRect::Radii(ColorSwatchCornerRadius * pageScaleFactor) };

    auto render = adoptNS([PAL::allocUIGraphicsImageRendererInstance() initWithSize:imageRect.size()]);
    UIImage *image = [render imageWithActions:^(UIGraphicsImageRendererContext *rendererContext) {
        GraphicsContextCG context { rendererContext.CGContext };
        context.translate(0, CGRectGetHeight(imageRect));
        context.scale({ 1, -1 });
        context.fillRoundedRect(swatch, color);
    }];

    visiblePath.addRoundedRect(swatch);
    return image.CGImage;
}

#else

void deleteDragImage(RetainPtr<CGImageRef>)
{
    // Since this is a RetainPtr, there's nothing additional we need to do to
    // delete it. It will be released when it falls out of scope.
}

// FIXME: fix signature of dragImageSize() to avoid copying the argument.
IntSize dragImageSize(RetainPtr<CGImageRef> image)
{
    return IntSize(CGImageGetWidth(image.get()), CGImageGetHeight(image.get()));
}

RetainPtr<CGImageRef> scaleDragImage(RetainPtr<CGImageRef>, FloatSize)
{
    return nullptr;
}

RetainPtr<CGImageRef> createDragImageFromImage(Image*, ImageOrientation)
{
    return nullptr;
}

DragImageRef createDragImageForRange(LocalFrame&, const SimpleRange&, bool)
{
    return nullptr;
}

#endif

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY)
