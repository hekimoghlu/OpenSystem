/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 24, 2025.
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
#include "WKImageCG.h"

#include "WKSharedAPICast.h"
#include "WKString.h"
#include "WebImage.h"
#include <WebCore/ColorSpace.h>
#include <WebCore/GraphicsContext.h>
#include <WebCore/ImageBufferUtilitiesCG.h>
#include <WebCore/NativeImage.h>
#include <WebCore/ShareableBitmap.h>

CGImageRef WKImageCreateCGImage(WKImageRef imageRef)
{
    WebKit::WebImage* webImage = WebKit::toImpl(imageRef);
    if (!webImage)
        return nullptr;

    auto nativeImage = webImage->copyNativeImage();
    if (!nativeImage)
        return nullptr;

    auto platformImage = nativeImage->platformImage();
    return platformImage.leakRef();
}

WKImageRef WKImageCreateFromCGImage(CGImageRef imageRef, WKImageOptions options)
{
    if (!imageRef)
        return nullptr;
    
    auto nativeImage = WebCore::NativeImage::create(imageRef);
    WebCore::IntSize imageSize = nativeImage->size();
    auto webImage = WebKit::WebImage::create(imageSize, WebKit::toImageOptions(options), WebCore::DestinationColorSpace::SRGB());
    if (!webImage->context())
        return nullptr;
    auto& graphicsContext = *webImage->context();

    WebCore::FloatRect rect(WebCore::FloatPoint(0, 0), imageSize);

    graphicsContext.clearRect(rect);
    graphicsContext.drawNativeImage(*nativeImage, rect, rect);
    return toAPI(webImage.leakRef());
}

WKStringRef WKImageCreateDataURLFromImage(CGImageRef imageRef)
{
    String mimeType { "image/png"_s };
    auto value = WebCore::dataURL(imageRef, mimeType, { });
    return WKStringCreateWithUTF8CString(value.utf8().data());
}
