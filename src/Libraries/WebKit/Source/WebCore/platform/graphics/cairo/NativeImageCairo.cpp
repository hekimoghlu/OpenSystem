/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 5, 2024.
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
#include "NativeImage.h"

#if USE(CAIRO)

#include "CairoOperations.h"
#include "CairoUtilities.h"
#include "NotImplemented.h"
#include <cairo.h>

namespace WebCore {

IntSize PlatformImageNativeImageBackend::size() const
{
    return cairoSurfaceSize(m_platformImage.get());
}

bool PlatformImageNativeImageBackend::hasAlpha() const
{
    return cairo_surface_get_content(m_platformImage.get()) != CAIRO_CONTENT_COLOR;
}

DestinationColorSpace PlatformImageNativeImageBackend::colorSpace() const
{
    notImplemented();
    return DestinationColorSpace::SRGB();
}

Headroom PlatformImageNativeImageBackend::headroom() const
{
    return Headroom::None;
}

std::optional<Color> NativeImage::singlePixelSolidColor() const
{
    if (size() != IntSize(1, 1))
        return std::nullopt;

    auto platformImage = this->platformImage().get();
    if (cairo_surface_get_type(platformImage) != CAIRO_SURFACE_TYPE_IMAGE)
        return std::nullopt;

    unsigned* pixel = reinterpret_cast_ptr<unsigned*>(cairo_image_surface_get_data(platformImage));
    return unpremultiplied(asSRGBA(PackedColor::ARGB { *pixel }));
}

void NativeImage::draw(GraphicsContext& context, const FloatRect& destinationRect, const FloatRect& sourceRect, ImagePaintingOptions options)
{
    context.drawNativeImageInternal(*this, destinationRect, sourceRect, options);
}

void NativeImage::clearSubimages()
{
}

#if USE(COORDINATED_GRAPHICS)
uint64_t NativeImage::uniqueID() const
{
    if (auto& image = platformImage())
        return getSurfaceUniqueID(image.get());
    return 0;
}
#endif

} // namespace WebCore

#endif // USE(CAIRO)
