/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 9, 2021.
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
#include "MediaPlayerPrivateMediaFoundation.h"

#if ENABLE(VIDEO) && USE(MEDIA_FOUNDATION) && USE(CAIRO)

#include "CairoOperations.h"

namespace WebCore {

void MediaPlayerPrivateMediaFoundation::Direct3DPresenter::paintCurrentFrame(GraphicsContext& context, const FloatRect& destRect)
{
    UINT width = m_destRect.right - m_destRect.left;
    UINT height = m_destRect.bottom - m_destRect.top;

    if (!width || !height)
        return;

    Locker locker { m_lock };

    if (!m_memSurface)
        return;

    D3DLOCKED_RECT lockedRect;
    if (SUCCEEDED(m_memSurface->LockRect(&lockedRect, nullptr, D3DLOCK_READONLY))) {
        void* data = lockedRect.pBits;
        int pitch = lockedRect.Pitch;
        D3DFORMAT format = D3DFMT_UNKNOWN;
        D3DSURFACE_DESC desc;
        if (SUCCEEDED(m_memSurface->GetDesc(&desc)))
            format = desc.Format;

        cairo_format_t cairoFormat = CAIRO_FORMAT_INVALID;

        switch (format) {
        case D3DFMT_A8R8G8B8:
            cairoFormat = CAIRO_FORMAT_ARGB32;
            break;
        case D3DFMT_X8R8G8B8:
            cairoFormat = CAIRO_FORMAT_RGB24;
            break;
        default:
            break;
        }

        ASSERT(cairoFormat != CAIRO_FORMAT_INVALID);

        if (cairoFormat != CAIRO_FORMAT_INVALID) {
            auto surface = adoptRef(cairo_image_surface_create_for_data(static_cast<unsigned char*>(data), cairoFormat, width, height, pitch));
            auto image = NativeImage::create(WTFMove(surface));
            FloatRect srcRect(0, 0, width, height);
            context.drawNativeImage(*image, destRect, srcRect);
        }
        m_memSurface->UnlockRect();
    }
}

} // namespace WebCore

#endif // ENABLE(VIDEO) && USE(MEDIA_FOUNDATION) && USE(CAIRO)
