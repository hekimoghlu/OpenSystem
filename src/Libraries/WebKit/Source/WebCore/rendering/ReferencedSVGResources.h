/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 14, 2025.
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

#include "SVGNames.h"
#include <wtf/RobinHoodHashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/AtomString.h>
#include <wtf/text/AtomStringHash.h>

namespace WebCore {

class CSSSVGResourceElementClient;
class Document;
class LegacyRenderSVGResourceClipper;
class LegacyRenderSVGResourceContainer;
class QualifiedName;
class ReferenceFilterOperation;
class ReferencePathOperation;
class RenderElement;
class RenderSVGResourceFilter;
class RenderStyle;
class SVGClipPathElement;
class SVGElement;
class SVGFilterElement;
class SVGMarkerElement;
class SVGMaskElement;
class StyleImage;
class TreeScope;

class ReferencedSVGResources {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ReferencedSVGResources);
public:
    ReferencedSVGResources(RenderElement&);
    ~ReferencedSVGResources();

    using SVGQualifiedNames = Vector<SVGQualifiedName>;
    using SVGElementIdentifierAndTagPairs = Vector<std::pair<AtomString, SVGQualifiedNames>>;

    static SVGElementIdentifierAndTagPairs referencedSVGResourceIDs(const RenderStyle&, const Document&);
    void updateReferencedResources(TreeScope&, const SVGElementIdentifierAndTagPairs&);

    // Legacy: Clipping needs a renderer, filters use an element.
    static LegacyRenderSVGResourceClipper* referencedClipperRenderer(TreeScope&, const ReferencePathOperation&);
    static RefPtr<SVGFilterElement> referencedFilterElement(TreeScope&, const ReferenceFilterOperation&);

    static LegacyRenderSVGResourceContainer* referencedRenderResource(TreeScope&, const AtomString& fragment);

    // LBSE: All element based.
    static RefPtr<SVGClipPathElement> referencedClipPathElement(TreeScope&, const ReferencePathOperation&);
    static RefPtr<SVGMarkerElement> referencedMarkerElement(TreeScope&, const String&);
    static RefPtr<SVGMaskElement> referencedMaskElement(TreeScope&, const StyleImage&);
    static RefPtr<SVGMaskElement> referencedMaskElement(TreeScope&, const AtomString&);
    static RefPtr<SVGElement> referencedPaintServerElement(TreeScope&, const String&);

private:
    static RefPtr<SVGElement> elementForResourceID(TreeScope&, const AtomString& resourceID, const SVGQualifiedName& tagName);
    static RefPtr<SVGElement> elementForResourceIDs(TreeScope&, const AtomString& resourceID, const SVGQualifiedNames& tagNames);

    void addClientForTarget(SVGElement& targetElement, const AtomString&);
    void removeClientForTarget(TreeScope&, const AtomString&);

    CheckedRef<RenderElement> m_renderer;
    MemoryCompactRobinHoodHashMap<AtomString, std::unique_ptr<CSSSVGResourceElementClient>> m_elementClients;
};

} // namespace WebCore
