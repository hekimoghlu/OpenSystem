/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 2, 2025.
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
#include "RenderSVGTextPath.h"

#include "FloatQuad.h"
#include "RenderBlock.h"
#include "RenderBoxModelObjectInlines.h"
#include "RenderLayer.h"
#include "RenderSVGInlineInlines.h"
#include "RenderSVGShape.h"
#include "SVGElementTypeHelpers.h"
#include "SVGGeometryElement.h"
#include "SVGInlineTextBox.h"
#include "SVGNames.h"
#include "SVGPathData.h"
#include "SVGPathElement.h"
#include "SVGRootInlineBox.h"
#include "SVGTextPathElement.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderSVGTextPath);

RenderSVGTextPath::RenderSVGTextPath(SVGTextPathElement& element, RenderStyle&& style)
    : RenderSVGInline(Type::SVGTextPath, element, WTFMove(style))
{
    ASSERT(isRenderSVGTextPath());
}

RenderSVGTextPath::~RenderSVGTextPath() = default;

SVGTextPathElement& RenderSVGTextPath::textPathElement() const
{
    return downcast<SVGTextPathElement>(RenderSVGInline::graphicsElement());
}

SVGGeometryElement* RenderSVGTextPath::targetElement() const
{
    auto target = SVGURIReference::targetElementFromIRIString(textPathElement().href(), textPathElement().treeScopeForSVGReferences());
    return dynamicDowncast<SVGGeometryElement>(target.element.get());
}

Path RenderSVGTextPath::layoutPath() const
{
    RefPtr element = targetElement();
    if (!is<SVGGeometryElement>(element))
        return { };

    auto path = pathFromGraphicsElement(*element);

    // Spec:  The transform attribute on the referenced 'path' element represents a
    // supplemental transformation relative to the current user coordinate system for
    // the current 'text' element, including any adjustments to the current user coordinate
    // system due to a possible transform attribute on the current 'text' element.
    // http://www.w3.org/TR/SVG/text.html#TextPathElement
    if (element->renderer() && document().settings().layerBasedSVGEngineEnabled()) {
        auto& renderer = downcast<RenderSVGShape>(*element->renderer());
        if (auto* layer = renderer.layer()) {
            const auto& layerTransform = layer->currentTransform(RenderStyle::individualTransformOperations()).toAffineTransform();
            if (!layerTransform.isIdentity())
                path.transform(layerTransform);
            return path;
        }
    }

    path.transform(element->animatedLocalTransform());
    return path;
}

const SVGLengthValue& RenderSVGTextPath::startOffset() const
{
    return textPathElement().startOffset();
}

}
