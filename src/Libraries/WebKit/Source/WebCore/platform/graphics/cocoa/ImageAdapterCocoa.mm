/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 12, 2022.
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
#import "ImageAdapter.h"

#import "BitmapImage.h"
#import "FloatRect.h"
#import "GraphicsContext.h"
#import "SharedBuffer.h"
#import <wtf/cocoa/SpanCocoa.h>
#import <wtf/text/WTFString.h>

#if ENABLE(MULTI_REPRESENTATION_HEIC)
#import "PlatformNSAdaptiveImageGlyph.h"
#endif

#if PLATFORM(IOS_FAMILY)
#import "UIFoundationSoftLink.h"
#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>
#import <MobileCoreServices/MobileCoreServices.h>
#endif

@interface WebCoreBundleFinder : NSObject
@end

@implementation WebCoreBundleFinder
@end

namespace WebCore {

Ref<Image> ImageAdapter::loadPlatformResource(const char *name)
{
    NSBundle *bundle = [NSBundle bundleForClass:[WebCoreBundleFinder class]];
    NSString *imagePath = [bundle pathForResource:[NSString stringWithUTF8String:name] ofType:@"png"];
    NSData *namedImageData = [NSData dataWithContentsOfFile:imagePath];
    if (namedImageData) {
        auto image = BitmapImage::create();
        image->setData(SharedBuffer::create(namedImageData), true);
        return WTFMove(image);
    }

    // We have reports indicating resource loads are failing, but we don't yet know the root cause(s).
    // Two theories are bad installs (image files are missing), and too-many-open-files.
    // See rdar://5607381
    ASSERT_NOT_REACHED();
    return Image::nullImage();
}

RetainPtr<CFDataRef> ImageAdapter::tiffRepresentation(const Vector<Ref<NativeImage>>& nativeImages)
{
    // If nativeImages.size() is zero, we know for certain this image doesn't have valid data
    // Even though the call to CGImageDestinationCreateWithData will fail and we'll handle it gracefully,
    // in certain circumstances that call will spam the console with an error message
    if (!nativeImages.size())
        return nullptr;

    RetainPtr<CFMutableDataRef> data = adoptCF(CFDataCreateMutable(0, 0));

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    RetainPtr<CGImageDestinationRef> destination = adoptCF(CGImageDestinationCreateWithData(data.get(), kUTTypeTIFF, nativeImages.size(), 0));
ALLOW_DEPRECATED_DECLARATIONS_END

    if (!destination)
        return nullptr;

    for (const auto& nativeImage : nativeImages)
        CGImageDestinationAddImage(destination.get(), nativeImage->platformImage().get(), 0);

    CGImageDestinationFinalize(destination.get());
    return data;
}

#if ENABLE(MULTI_REPRESENTATION_HEIC)
NSAdaptiveImageGlyph *ImageAdapter::multiRepresentationHEIC()
{
    if (m_multiRepHEIC)
        return m_multiRepHEIC.get();

    auto buffer = image().data();
    if (!buffer)
        return nullptr;

    Vector<uint8_t> data = buffer->copyData();

    RetainPtr nsData = toNSData(data.span());
    m_multiRepHEIC = adoptNS([[PlatformNSAdaptiveImageGlyph alloc] initWithImageContent:nsData.get()]);

    return m_multiRepHEIC.get();
}
#endif

void ImageAdapter::invalidate()
{
#if USE(APPKIT)
    m_nsImage = nullptr;
#endif
    m_tiffRep = nullptr;
#if ENABLE(MULTI_REPRESENTATION_HEIC)
    m_multiRepHEIC = nullptr;
#endif
}

CFDataRef ImageAdapter::tiffRepresentation()
{
    if (m_tiffRep)
        return m_tiffRep.get();

    auto data = tiffRepresentation(allNativeImages());
    if (!data)
        return nullptr;

    m_tiffRep = data;
    return m_tiffRep.get();
}

#if USE(APPKIT)
NSImage* ImageAdapter::nsImage()
{
    if (m_nsImage)
        return m_nsImage.get();

    CFDataRef data = tiffRepresentation();
    if (!data)
        return nullptr;

    m_nsImage = adoptNS([[NSImage alloc] initWithData:(__bridge NSData *)data]);
    return m_nsImage.get();
}

RetainPtr<NSImage> ImageAdapter::snapshotNSImage()
{
    RefPtr nativeImage =  image().currentNativeImage();
    if (!nativeImage)
        return nullptr;

    auto data = tiffRepresentation({ nativeImage.releaseNonNull() });
    if (!data)
        return nullptr;

    return adoptNS([[NSImage alloc] initWithData:(__bridge NSData *)data.get()]);
}
#endif

} // namespace WebCore
