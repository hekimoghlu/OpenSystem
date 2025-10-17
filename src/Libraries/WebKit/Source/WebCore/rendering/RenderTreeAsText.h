/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 17, 2022.
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

#include <wtf/Forward.h>
#include <wtf/OptionSet.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

class Element;
class LocalFrame;
class RenderObject;
class RenderView;

enum class RenderAsTextFlag : uint16_t {
    ShowAllLayers           = 1 << 0, // Dump all layers, not just those that would paint.
    ShowLayerNesting        = 1 << 1, // Annotate the layer lists.
    ShowCompositedLayers    = 1 << 2, // Show which layers are composited.
    ShowOverflow            = 1 << 3, // Print layout and visual overflow.
    ShowSVGGeometry         = 1 << 4, // Print additional geometry for SVG objects.
    ShowLayerFragments      = 1 << 5, // Print additional info about fragmented RenderLayers
    ShowAddresses           = 1 << 6, // Show layer and renderer addresses.
    ShowIDAndClass          = 1 << 7, // Show id and class attributes
    PrintingMode            = 1 << 8, // Dump the tree in printing mode.
    DontUpdateLayout        = 1 << 9, // Don't update layout, to make it safe to call showLayerTree() from the debugger inside layout or painting code.
    ShowLayoutState         = 1 << 10, // Print the various 'needs layout' bits on renderers.
};

// You don't need pageWidthInPixels if you don't specify RenderAsTextInPrintingMode.
WEBCORE_EXPORT TextStream createTextStream(const RenderView&);
WEBCORE_EXPORT String externalRepresentation(LocalFrame*, OptionSet<RenderAsTextFlag> = { });
WEBCORE_EXPORT String externalRepresentation(Element*, OptionSet<RenderAsTextFlag> = { });
WEBCORE_EXPORT void externalRepresentationForLocalFrame(TextStream&, LocalFrame&, OptionSet<RenderAsTextFlag> = { });
void write(WTF::TextStream&, const RenderObject&, OptionSet<RenderAsTextFlag> = { });
void writeDebugInfo(WTF::TextStream&, const RenderObject&, OptionSet<RenderAsTextFlag> = { });

class RenderTreeAsText {
// FIXME: This is a cheesy hack to allow easy access to RenderStyle colors.  It won't be needed if we convert
// it to use visitedDependentColor instead. (This just involves rebaselining many results though, so for now it's
// not being done).
public:
    static void writeRenderObject(WTF::TextStream&, const RenderObject&, OptionSet<RenderAsTextFlag>);
};

// Helper function shared with SVGRenderTreeAsText
String quoteAndEscapeNonPrintables(StringView);

WEBCORE_EXPORT String counterValueForElement(Element*);
WEBCORE_EXPORT String markerTextForListItem(Element*);

} // namespace WebCore
