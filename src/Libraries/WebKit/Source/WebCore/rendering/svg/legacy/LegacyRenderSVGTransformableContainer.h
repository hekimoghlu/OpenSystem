/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 3, 2023.
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

#include "LegacyRenderSVGContainer.h"

namespace WebCore {
    
class SVGGraphicsElement;

class LegacyRenderSVGTransformableContainer final : public LegacyRenderSVGContainer {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(LegacyRenderSVGTransformableContainer);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(LegacyRenderSVGTransformableContainer);
public:
    LegacyRenderSVGTransformableContainer(SVGGraphicsElement&, RenderStyle&&);
    virtual ~LegacyRenderSVGTransformableContainer();

    const AffineTransform& localToParentTransform() const override { return m_localTransform; }
    void setNeedsTransformUpdate() override { m_needsTransformUpdate = true; }
    bool didTransformToRootUpdate() override { return m_didTransformToRootUpdate; }

private:
    SVGGraphicsElement& graphicsElement();

    void element() const = delete;
    bool calculateLocalTransform() override;
    AffineTransform localTransform() const override { return m_localTransform; }

    bool m_needsTransformUpdate : 1;
    bool m_didTransformToRootUpdate : 1;
    AffineTransform m_localTransform;
    FloatSize m_lastTranslation;
    FloatRect m_lastTransformReferenceBoxRect;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(LegacyRenderSVGTransformableContainer, isLegacyRenderSVGTransformableContainer())
