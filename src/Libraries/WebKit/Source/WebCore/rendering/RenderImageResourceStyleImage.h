/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 16, 2025.
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

#include "RenderImageResource.h"
#include "StyleImage.h"
#include <wtf/RefPtr.h>

namespace WebCore {

class RenderElement;

class RenderImageResourceStyleImage final : public RenderImageResource {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderImageResourceStyleImage);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderImageResourceStyleImage);
public:
    explicit RenderImageResourceStyleImage(StyleImage&);

private:
    void initialize(RenderElement&) final;
    void shutdown() final;

    RefPtr<Image> image(const IntSize& = { }) const final;
    bool errorOccurred() const final { return m_styleImage->errorOccurred(); }

    void setContainerContext(const IntSize&, const URL&) final;

    bool imageHasRelativeWidth() const final { return m_styleImage->imageHasRelativeWidth(); }
    bool imageHasRelativeHeight() const final { return m_styleImage->imageHasRelativeHeight(); }

    WrappedImagePtr imagePtr() const final { return m_styleImage->data(); }
    LayoutSize imageSize(float multiplier, CachedImage::SizeType) const final { return LayoutSize(m_styleImage->imageSize(renderer(), multiplier)); }

    Ref<StyleImage> m_styleImage;
};

} // namespace WebCore
