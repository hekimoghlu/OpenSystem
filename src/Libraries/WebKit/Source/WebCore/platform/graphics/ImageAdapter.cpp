/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 20, 2024.
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
#include "ImageAdapter.h"

#include "BitmapImage.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ImageAdapter);

#if !PLATFORM(COCOA) && !PLATFORM(GTK) && !PLATFORM(WIN)
Ref<Image> ImageAdapter::loadPlatformResource(const char* resource)
{
    WTFLogAlways("WARNING: trying to load platform resource '%s'", resource);
    return BitmapImage::create();
}

void ImageAdapter::invalidate()
{
}
#endif // !PLATFORM(COCOA) && !PLATFORM(GTK) && !PLATFORM(WIN)

RefPtr<NativeImage> ImageAdapter::nativeImageOfSize(const IntSize& size)
{
    unsigned count = image().frameCount();

    for (unsigned i = 0; i < count; ++i) {
        RefPtr nativeImage = image().nativeImageAtIndex(i);
        if (nativeImage && nativeImage->size() == size)
            return nativeImage;
    }

    // Fallback to the first frame image if we can't find the right size
    return image().nativeImageAtIndex(0);
}

Vector<Ref<NativeImage>> ImageAdapter::allNativeImages()
{
    Vector<Ref<NativeImage>> nativeImages;
    unsigned count = image().frameCount();

    for (unsigned i = 0; i < count; ++i) {
        if (RefPtr nativeImage = image().nativeImageAtIndex(i))
            nativeImages.append(nativeImage.releaseNonNull());
    }

    return nativeImages;
}

} // namespace WebCore
