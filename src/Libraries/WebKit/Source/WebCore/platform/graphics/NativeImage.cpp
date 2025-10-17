/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 19, 2024.
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

#if USE(SKIA)
#include "GLFence.h"
#endif

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(NativeImage);

NativeImageBackend::NativeImageBackend() = default;

NativeImageBackend::~NativeImageBackend() = default;

bool NativeImageBackend::isRemoteNativeImageBackendProxy() const
{
    return false;
}

PlatformImageNativeImageBackend::~PlatformImageNativeImageBackend() = default;

const PlatformImagePtr& PlatformImageNativeImageBackend::platformImage() const
{
    return m_platformImage;
}

PlatformImageNativeImageBackend::PlatformImageNativeImageBackend(PlatformImagePtr platformImage)
    : m_platformImage(WTFMove(platformImage))
{
}

#if !USE(CG)
RefPtr<NativeImage> NativeImage::create(PlatformImagePtr&& platformImage, RenderingResourceIdentifier identifier)
{
    if (!platformImage)
        return nullptr;
    UniqueRef<PlatformImageNativeImageBackend> backend { *new PlatformImageNativeImageBackend(WTFMove(platformImage)) };
    return adoptRef(*new NativeImage(WTFMove(backend), identifier));
}

RefPtr<NativeImage> NativeImage::createTransient(PlatformImagePtr&& image, RenderingResourceIdentifier identifier)
{
    return create(WTFMove(image), identifier);
}
#endif

NativeImage::NativeImage(UniqueRef<NativeImageBackend> backend, RenderingResourceIdentifier renderingResourceIdentifier)
    : RenderingResource(renderingResourceIdentifier)
    , m_backend(WTFMove(backend))
{
}

NativeImage::~NativeImage() = default;

const PlatformImagePtr& NativeImage::platformImage() const
{
    return m_backend->platformImage();
}

IntSize NativeImage::size() const
{
    return m_backend->size();
}

bool NativeImage::hasAlpha() const
{
    return m_backend->hasAlpha();
}

DestinationColorSpace NativeImage::colorSpace() const
{
    return m_backend->colorSpace();
}

Headroom NativeImage::headroom() const
{
    return m_backend->headroom();
}

void NativeImage::replaceBackend(UniqueRef<NativeImageBackend> backend)
{
    m_backend = WTFMove(backend);
}

} // namespace WebCore
