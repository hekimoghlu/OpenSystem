/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 25, 2024.
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
#include "GraphicsContext.h"

#include "AffineTransform.h"
#include "BitmapInfo.h"
#include "TransformationMatrix.h"
#include "NotImplemented.h"
#include "Path.h"
#include <wtf/MathExtras.h>
#include <wtf/win/GDIObject.h>

namespace WebCore {

static void fillWithClearColor(HBITMAP bitmap)
{
    BITMAP bmpInfo;
    GetObject(bitmap, sizeof(bmpInfo), &bmpInfo);
    int bufferSize = bmpInfo.bmWidthBytes * bmpInfo.bmHeight;
    memset(bmpInfo.bmBits, 0, bufferSize);
}

HDC GraphicsContext::getWindowsContext(const IntRect& dstRect, bool supportAlphaBlend)
{
    if (!hasPlatformContext())
        return nullptr;
    HDC hdc = nullptr;
    // FIXME: Should a bitmap be created also when a shadow is set?
    if (dstRect.isEmpty())
        return 0;

    // Create a bitmap DC in which to draw.
    BitmapInfo bitmapInfo = BitmapInfo::create(dstRect.size());

    void* pixels = 0;
    HBITMAP bitmap = ::CreateDIBSection(nullptr, &bitmapInfo, DIB_RGB_COLORS, &pixels, 0, 0);
    if (!bitmap)
        return 0;

    auto bitmapDC = adoptGDIObject(::CreateCompatibleDC(hdc));
    ::SelectObject(bitmapDC.get(), bitmap);

    // Fill our buffer with clear if we're going to alpha blend.
    if (supportAlphaBlend)
        fillWithClearColor(bitmap);

    // Make sure we can do world transforms.
    ::SetGraphicsMode(bitmapDC.get(), GM_ADVANCED);

    // Apply a translation to our context so that the drawing done will be at (0,0) of the bitmap.
    XFORM xform = TransformationMatrix().translate(-dstRect.x(), -dstRect.y());

    ::SetWorldTransform(bitmapDC.get(), &xform);

    return bitmapDC.leak();
}

#if USE(SKIA)
void GraphicsContext::releaseWindowsContext(HDC, const IntRect&, bool)
{
    notImplemented();
}
#endif

} // namespace WebCore
