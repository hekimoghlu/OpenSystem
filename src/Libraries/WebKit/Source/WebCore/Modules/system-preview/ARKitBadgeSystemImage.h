/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 2, 2022.
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
#pragma once

#if USE(SYSTEM_PREVIEW)

#include "Image.h"
#include "NativeImage.h"
#include "SystemImage.h"
#include <optional>
#include <wtf/ArgumentCoder.h>
#include <wtf/Forward.h>
#include <wtf/Ref.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>

OBJC_CLASS CIContext;

namespace WebCore {

class WEBCORE_EXPORT ARKitBadgeSystemImage final : public SystemImage {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ARKitBadgeSystemImage, WEBCORE_EXPORT);
public:
    static Ref<ARKitBadgeSystemImage> create(Image& image)
    {
        return adoptRef(*new ARKitBadgeSystemImage(image));
    }

    static Ref<ARKitBadgeSystemImage> create(RenderingResourceIdentifier renderingResourceIdentifier, FloatSize size)
    {
        return adoptRef(*new ARKitBadgeSystemImage(renderingResourceIdentifier, size));
    }

    virtual ~ARKitBadgeSystemImage() = default;

    void draw(GraphicsContext&, const FloatRect&) const final;

    Image* image() const { return m_image.get(); }
    void setImage(Image& image) { m_image = &image; }

    RenderingResourceIdentifier imageIdentifier() const;

private:
    friend struct IPC::ArgumentCoder<ARKitBadgeSystemImage, void>;
    ARKitBadgeSystemImage(Image& image)
        : SystemImage(SystemImageType::ARKitBadge)
        , m_image(&image)
        , m_imageSize(image.size())
    {
    }

    ARKitBadgeSystemImage(RenderingResourceIdentifier renderingResourceIdentifier, FloatSize size)
        : SystemImage(SystemImageType::ARKitBadge)
        , m_renderingResourceIdentifier(renderingResourceIdentifier)
        , m_imageSize(size)
    {
    }

    RefPtr<Image> m_image;
    Markable<RenderingResourceIdentifier> m_renderingResourceIdentifier;
    FloatSize m_imageSize;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ARKitBadgeSystemImage)
    static bool isType(const WebCore::SystemImage& systemImage) { return systemImage.systemImageType() == WebCore::SystemImageType::ARKitBadge; }
SPECIALIZE_TYPE_TRAITS_END()

#endif // USE(SYSTEM_PREVIEW)
