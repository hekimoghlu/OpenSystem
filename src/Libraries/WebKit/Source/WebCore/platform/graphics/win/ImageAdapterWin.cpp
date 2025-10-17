/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 1, 2022.
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

#if PLATFORM(WIN)

#include "BitmapImage.h"
#include "SharedBuffer.h"
#include "WebCoreBundleWin.h"
#include <windows.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

Ref<Image> ImageAdapter::loadPlatformResource(const char *name)
{
    auto path = webKitBundlePath(StringView::fromLatin1(name), "png"_s, "icons"_s);
    auto data = FileSystem::readEntireFile(path);
    auto img = BitmapImage::create();
    if (data)
        img->setData(FragmentedSharedBuffer::create(WTFMove(*data)), true);
    return img;
}

void ImageAdapter::invalidate()
{
}

bool ImageAdapter::getHBITMAP(HBITMAP bmp)
{
    return getHBITMAPOfSize(bmp, 0);
}

#if USE(SKIA)
bool ImageAdapter::getHBITMAPOfSize(HBITMAP, const IntSize*)
{
    return false;
}

RefPtr<NativeImage> ImageAdapter::nativeImageOfHBITMAP(HBITMAP)
{
    return nullptr;
}
#endif

} // namespace WebCore

#endif // PLATFORM(WIN)
