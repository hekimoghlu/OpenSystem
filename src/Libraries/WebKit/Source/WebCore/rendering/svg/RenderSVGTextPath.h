/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 12, 2023.
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

#include "Path.h"
#include "RenderSVGInline.h"

namespace WebCore {

class SVGGeometryElement;
class SVGTextPathElement;

class RenderSVGTextPath final : public RenderSVGInline {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderSVGTextPath);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderSVGTextPath);
public:
    RenderSVGTextPath(SVGTextPathElement&, RenderStyle&&);
    virtual ~RenderSVGTextPath();

    SVGTextPathElement& textPathElement() const;
    SVGGeometryElement* targetElement() const;

    Path layoutPath() const;
    const SVGLengthValue& startOffset() const;

private:
    void graphicsElement() const = delete;

    ASCIILiteral renderName() const override { return "RenderSVGTextPath"_s; }

    Path m_layoutPath;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderSVGTextPath, isRenderSVGTextPath())
