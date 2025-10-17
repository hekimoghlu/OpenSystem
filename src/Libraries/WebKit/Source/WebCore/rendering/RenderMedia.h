/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 19, 2025.
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

#if ENABLE(VIDEO)

#include "HTMLMediaElement.h"
#include "RenderImage.h"

namespace WebCore {

class RenderMedia : public RenderImage {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderMedia);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderMedia);
public:
    RenderMedia(Type, HTMLMediaElement&, RenderStyle&&);
    RenderMedia(Type, HTMLMediaElement&, RenderStyle&&, const IntSize& intrinsicSize);
    virtual ~RenderMedia();

    HTMLMediaElement& mediaElement() const { return downcast<HTMLMediaElement>(nodeForNonAnonymous()); }

    bool shouldDisplayBrokenImageIcon() const final { return false; }

protected:
    void layout() override;
    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) override;

    void visibleInViewportStateChanged() override { }

private:
    void element() const = delete;

    bool canHaveChildren() const final { return true; }

    ASCIILiteral renderName() const override { return "RenderMedia"_s; }
    bool isImage() const final { return false; }
    void paintReplaced(PaintInfo&, const LayoutPoint&) override;
};

inline RenderMedia* HTMLMediaElement::renderer() const
{
    return downcast<RenderMedia>(HTMLElement::renderer());
}

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderMedia, isRenderMedia())

#endif // ENABLE(VIDEO)
