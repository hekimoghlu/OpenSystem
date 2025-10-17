/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 3, 2024.
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

#include "RenderTreeAsText.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

class LegacyRenderSVGContainer;
class LegacyRenderSVGImage;
class LegacyRenderSVGResourceContainer;
class LegacyRenderSVGRoot;
class LegacyRenderSVGShape;
class RenderElement;
class RenderObject;
class RenderSVGGradientStop;
class RenderSVGInlineText;
class RenderSVGText;
class SVGGraphicsElement;

// functions used by the main RenderTreeAsText code
void write(WTF::TextStream&, const LegacyRenderSVGRoot&, OptionSet<RenderAsTextFlag>);
void write(WTF::TextStream&, const LegacyRenderSVGShape&, OptionSet<RenderAsTextFlag>);
void writeSVGGradientStop(WTF::TextStream&, const RenderSVGGradientStop&, OptionSet<RenderAsTextFlag>);
void writeSVGResourceContainer(WTF::TextStream&, const LegacyRenderSVGResourceContainer&, OptionSet<RenderAsTextFlag>);
void writeSVGContainer(WTF::TextStream&, const LegacyRenderSVGContainer&, OptionSet<RenderAsTextFlag>);
void writeSVGGraphicsElement(WTF::TextStream&, const SVGGraphicsElement&);
void writeSVGImage(WTF::TextStream&, const LegacyRenderSVGImage&, OptionSet<RenderAsTextFlag>);
void writeSVGInlineText(WTF::TextStream&, const RenderSVGInlineText&, OptionSet<RenderAsTextFlag>);
void writeSVGPaintingFeatures(TextStream&, const RenderElement&, OptionSet<RenderAsTextFlag>);
void writeSVGText(WTF::TextStream&, const RenderSVGText&, OptionSet<RenderAsTextFlag>);
void writeResources(WTF::TextStream&, const RenderObject&, OptionSet<RenderAsTextFlag>);

// helper operators specific to dumping the render tree. these are used in various classes to dump the render tree
// these could be defined in separate namespace to avoid matching these generic signatures unintentionally.

template<typename Item>
WTF::TextStream& operator<<(WTF::TextStream& ts, const Vector<Item*>& v)
{
    ts << "[";

    for (unsigned i = 0; i < v.size(); i++) {
        ts << *v[i];
        if (i < v.size() - 1)
            ts << ", ";
    }

    ts << "]";
    return ts;
}

} // namespace WebCore
