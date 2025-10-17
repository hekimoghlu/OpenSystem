/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 22, 2022.
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
#import "FilterImage.h"

#if USE(CORE_IMAGE)

#import "ImageBufferIOSurfaceBackend.h"
#import <CoreImage/CIContext.h>
#import <CoreImage/CoreImage.h>
#import <wtf/NeverDestroyed.h>
#import <wtf/cocoa/TypeCastsCocoa.h>

namespace WebCore {

static RetainPtr<CIContext> sharedCIContext()
{
    static NeverDestroyed<RetainPtr<CIContext>> ciContext = [CIContext contextWithOptions:@{ kCIContextWorkingColorSpace: bridge_id_cast(adoptCF(CGColorSpaceCreateWithName(kCGColorSpaceSRGB))).get() }];
    return ciContext;
}

void FilterImage::setCIImage(RetainPtr<CIImage>&& ciImage)
{
    ASSERT(ciImage);
    m_ciImage = WTFMove(ciImage);
}

size_t FilterImage::memoryCostOfCIImage() const
{
    ASSERT(m_ciImage);
    return FloatSize([m_ciImage.get() extent].size).area() * 4;
}

ImageBuffer* FilterImage::imageBufferFromCIImage()
{
    ASSERT(m_ciImage);

    if (m_imageBuffer)
        return m_imageBuffer.get();

    m_imageBuffer = ImageBuffer::create<ImageBufferIOSurfaceBackend>(m_absoluteImageRect.size(), 1, m_colorSpace, ImageBufferPixelFormat::BGRA8, RenderingPurpose::Unspecified, { });
    if (!m_imageBuffer)
        return nullptr;

    ASSERT(m_imageBuffer->surface());
    auto destRect = FloatRect { FloatPoint(), m_absoluteImageRect.size() };
    [sharedCIContext().get() render:m_ciImage.get() toIOSurface:m_imageBuffer->surface()->surface() bounds:destRect colorSpace:m_colorSpace.platformColorSpace()];

    return m_imageBuffer.get();
}

} // namespace WebCore

#endif // USE(CORE_IMAGE)
