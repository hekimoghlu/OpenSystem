/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 29, 2023.
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
#import "Icon.h"

#if PLATFORM(COCOA)

#import "BitmapImage.h"
#import "GraphicsContext.h"
#import <pal/ios/UIKitSoftLink.h>

namespace WebCore {

Icon::Icon(CocoaImage *image)
    : m_image(image)
{
}

Icon::~Icon()
{
}

RefPtr<Icon> Icon::create(CocoaImage *image)
{
    if (!image)
        return nullptr;

    return adoptRef(new Icon(image));
}

RefPtr<Icon> Icon::create(PlatformImagePtr&& platformImage)
{
    if (!platformImage)
        return nullptr;

#if USE(APPKIT)
    auto image = adoptNS([[NSImage alloc] initWithCGImage:platformImage.get() size:NSZeroSize]);
#else
    auto image = adoptNS([PAL::allocUIImageInstance() initWithCGImage:platformImage.get()]);
#endif
    return adoptRef(new Icon(image.get()));
}

void Icon::paint(GraphicsContext& context, const FloatRect& destRect)
{
    if (context.paintingDisabled())
        return;

    GraphicsContextStateSaver stateSaver(context);

#if USE(APPKIT)
    auto cgImage = [m_image CGImageForProposedRect:nil context:nil hints:nil];
#else
    auto cgImage = [m_image CGImage];
#endif
    auto image = NativeImage::create(cgImage);

    FloatRect srcRect(FloatPoint::zero(), image->size());
    context.drawNativeImage(*image, destRect, srcRect, { InterpolationQuality::High });
}

}

#endif // PLATFORM(COCOA)
