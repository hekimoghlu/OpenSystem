/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 16, 2022.
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
#import "WebIconUtilities.h"

#if PLATFORM(COCOA)

#if PLATFORM(IOS_FAMILY)
#import "UIKitSPI.h"
#import <MobileCoreServices/MobileCoreServices.h>
#else
#import <CoreServices/CoreServices.h>
#endif

#import "CocoaImage.h"
#import <AVFoundation/AVFoundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreMedia/CoreMedia.h>
#import <ImageIO/ImageIO.h>
#import <WebCore/PlatformImage.h>
#import <wtf/MathExtras.h>
#import <wtf/RetainPtr.h>

#import <pal/cf/CoreMediaSoftLink.h>
#import <pal/cocoa/AVFoundationSoftLink.h>

namespace WebKit {

static const CGFloat iconSideLength = 100;

static CGRect squareCropRectForSize(CGSize size)
{
    CGFloat smallerSide = std::min(size.width, size.height);
    CGRect cropRect = CGRectMake(0, 0, smallerSide, smallerSide);

    if (size.width < size.height)
        cropRect.origin.y = std::round((size.height - smallerSide) / 2);
    else
        cropRect.origin.x = std::round((size.width - smallerSide) / 2);

    return cropRect;
}

static PlatformImagePtr squareImage(CGImageRef image)
{
    if (!image)
        return nil;

    CGSize imageSize = CGSizeMake(CGImageGetWidth(image), CGImageGetHeight(image));
    if (imageSize.width == imageSize.height)
        return image;

    CGRect squareCropRect = squareCropRectForSize(imageSize);
    return adoptCF(CGImageCreateWithImageInRect(image, squareCropRect));
}

static RetainPtr<CocoaImage> thumbnailSizedImageForImage(CGImageRef image)
{
    auto squaredImage = squareImage(image);
    if (!squaredImage)
        return nullptr;

    CGRect destinationRect = CGRectMake(0, 0, iconSideLength, iconSideLength);

    RetainPtr colorSpace = CGImageGetColorSpace(image);
    if (!CGColorSpaceSupportsOutput(colorSpace.get()))
        colorSpace = adoptCF(CGColorSpaceCreateWithName(kCGColorSpaceSRGB));

    auto context = adoptCF(CGBitmapContextCreate(nil, iconSideLength, iconSideLength, 8, 4 * iconSideLength, colorSpace.get(), kCGImageAlphaPremultipliedLast));

    CGContextSetInterpolationQuality(context.get(), kCGInterpolationHigh);
    CGContextDrawImage(context.get(), destinationRect, squaredImage.get());

    auto scaledImage = adoptCF(CGBitmapContextCreateImage(context.get()));

    auto thumbnailImage = scaledImage.get() ?: squaredImage.get();
#if USE(APPKIT)
    return adoptNS([[NSImage alloc] initWithCGImage:thumbnailImage size:NSZeroSize]);
#else
    return adoptNS([[UIImage alloc] initWithCGImage:thumbnailImage]);
#endif
}

RetainPtr<CocoaImage> fallbackIconForFile(NSURL *file)
{
    ASSERT_ARG(file, [file isFileURL]);

#if PLATFORM(MAC)
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    return [[NSWorkspace sharedWorkspace] iconForFileType:[@"." stringByAppendingString:file.pathExtension]];
ALLOW_DEPRECATED_DECLARATIONS_END
#else
    NSError *error = nil;
    if (![file checkResourceIsReachableAndReturnError:&error])
        return nil;
    UIDocumentInteractionController *interactionController = [UIDocumentInteractionController interactionControllerWithURL:file];
    if (![interactionController.icons count])
        return nil;
    return thumbnailSizedImageForImage(interactionController.icons[0].CGImage);
#endif
}

const CFStringRef kCGImageSourceEnableRestrictedDecoding = CFSTR("kCGImageSourceEnableRestrictedDecoding");

RetainPtr<CocoaImage> iconForImageFile(NSURL *file)
{
    ASSERT_ARG(file, [file isFileURL]);

    NSDictionary *options = @{
        (id)kCGImageSourceCreateThumbnailFromImageIfAbsent: @YES,
        (id)kCGImageSourceThumbnailMaxPixelSize: @(iconSideLength),
        (id)kCGImageSourceCreateThumbnailWithTransform: @YES,
        (id)kCGImageSourceEnableRestrictedDecoding: @YES
    };
    RetainPtr<CGImageSource> imageSource = adoptCF(CGImageSourceCreateWithURL((CFURLRef)file, 0));
    RetainPtr<CGImageRef> thumbnail = adoptCF(CGImageSourceCreateThumbnailAtIndex(imageSource.get(), 0, (CFDictionaryRef)options));
    if (!thumbnail) {
        LOG_ERROR("Error creating thumbnail image for image: %@", file);
        return fallbackIconForFile(file);
    }

    return thumbnailSizedImageForImage(thumbnail.get());
}

RetainPtr<CocoaImage> iconForVideoFile(NSURL *file)
{
    ASSERT_ARG(file, [file isFileURL]);

    RetainPtr<AVURLAsset> asset = adoptNS([PAL::allocAVURLAssetInstance() initWithURL:file options:nil]);
    RetainPtr<AVAssetImageGenerator> generator = adoptNS([PAL::allocAVAssetImageGeneratorInstance() initWithAsset:asset.get()]);
    [generator setAppliesPreferredTrackTransform:YES];

    NSError *error = nil;
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    RetainPtr<CGImageRef> imageRef = adoptCF([generator copyCGImageAtTime:PAL::kCMTimeZero actualTime:nil error:&error]);
ALLOW_DEPRECATED_DECLARATIONS_END
    if (!imageRef) {
        LOG_ERROR("Error creating image for video '%@': %@", file, error);
        return fallbackIconForFile(file);
    }

    return thumbnailSizedImageForImage(imageRef.get());
}

RetainPtr<CocoaImage> iconForFiles(const Vector<String>& filenames)
{
    if (!filenames.size())
        return nil;

    // FIXME: We should generate an icon showing multiple files here, if applicable. Currently, if there are multiple
    // files, we only use the first URL to generate an icon.
    NSURL *file = [NSURL fileURLWithPath:filenames[0] isDirectory:NO];
    if (!file)
        return nil;

    ASSERT_ARG(file, [file isFileURL]);

    NSString *fileExtension = file.pathExtension;
    if (!fileExtension.length)
        return nil;

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    RetainPtr<CFStringRef> fileUTI = adoptCF(UTTypeCreatePreferredIdentifierForTag(kUTTagClassFilenameExtension, (CFStringRef)fileExtension, 0));

    if (UTTypeConformsTo(fileUTI.get(), kUTTypeImage))
        return iconForImageFile(file);

    if (UTTypeConformsTo(fileUTI.get(), kUTTypeMovie))
        return iconForVideoFile(file);
ALLOW_DEPRECATED_DECLARATIONS_END

    return fallbackIconForFile(file);
}

}

#endif
