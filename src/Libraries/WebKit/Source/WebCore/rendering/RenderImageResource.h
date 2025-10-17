/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 24, 2023.
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

#include "CachedImage.h"
#include "CachedResourceHandle.h"
#include "StyleImage.h"
#include <wtf/CheckedPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class CachedImage;
class RenderElement;

class RenderImageResource : public CanMakeCheckedPtr<RenderImageResource> {
    WTF_MAKE_NONCOPYABLE(RenderImageResource);
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderImageResource);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderImageResource);
public:
    RenderImageResource();
    virtual ~RenderImageResource();

    virtual void initialize(RenderElement& renderer) { initialize(renderer, nullptr); }
    virtual void shutdown();

    void setCachedImage(CachedResourceHandle<CachedImage>&&);
    CachedImage* cachedImage() const { return m_cachedImage.get(); }

    void resetAnimation();

    virtual RefPtr<Image> image(const IntSize& size = { }) const;
    virtual bool errorOccurred() const { return m_cachedImage && m_cachedImage->errorOccurred(); }

    virtual void setContainerContext(const IntSize&, const URL&);

    virtual bool imageHasRelativeWidth() const { return m_cachedImage && m_cachedImage->imageHasRelativeWidth(); }
    virtual bool imageHasRelativeHeight() const { return m_cachedImage && m_cachedImage->imageHasRelativeHeight(); }

    inline LayoutSize imageSize(float multiplier) const { return imageSize(multiplier, CachedImage::UsedSize); }
    inline LayoutSize intrinsicSize(float multiplier) const { return imageSize(multiplier, CachedImage::IntrinsicSize); }

    virtual WrappedImagePtr imagePtr() const { return m_cachedImage.get(); }

protected:
    RenderElement* renderer() const { return m_renderer.get(); }
    void initialize(RenderElement&, CachedImage*);
    
private:
    virtual LayoutSize imageSize(float multiplier, CachedImage::SizeType) const;

    SingleThreadWeakPtr<RenderElement> m_renderer;
    CachedResourceHandle<CachedImage> m_cachedImage;
    bool m_cachedImageRemoveClientIsNeeded { true };
};

} // namespace WebCore
