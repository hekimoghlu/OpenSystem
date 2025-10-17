/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 22, 2021.
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

#include "RenderStyleConstants.h"
#include "TextSizeAdjustment.h"
#include <wtf/CheckedRef.h>
#include <wtf/OptionSet.h>

namespace WebCore {

class Document;
class Element;
class EventTarget;
class RenderStyle;
class SVGElement;
class Settings;

enum class AnimationImpact : uint8_t;

namespace Style {

class Update;

class Adjuster {
public:
    Adjuster(const Document&, const RenderStyle& parentStyle, const RenderStyle* parentBoxStyle, Element*);

    void adjust(RenderStyle&, const RenderStyle* userAgentAppearanceStyle) const;
    void adjustAnimatedStyle(RenderStyle&, OptionSet<AnimationImpact>) const;

    static void adjustVisibilityForPseudoElement(RenderStyle&, const Element& host);
    static void adjustSVGElementStyle(RenderStyle&, const SVGElement&);
    static bool adjustEventListenerRegionTypesForRootStyle(RenderStyle&, const Document&);
    static void propagateToDocumentElementAndInitialContainingBlock(Update&, const Document&);
    static std::unique_ptr<RenderStyle> restoreUsedDocumentElementStyleToComputed(const RenderStyle&);

#if ENABLE(TEXT_AUTOSIZING)
    struct AdjustmentForTextAutosizing {
        std::optional<float> newFontSize;
        std::optional<float> newLineHeight;
        std::optional<AutosizeStatus> newStatus;
        explicit operator bool() const { return newFontSize || newLineHeight || newStatus; }
    };
    static AdjustmentForTextAutosizing adjustmentForTextAutosizing(const RenderStyle&, const Element&);
    static bool adjustForTextAutosizing(RenderStyle&, AdjustmentForTextAutosizing);
    static bool adjustForTextAutosizing(RenderStyle&, const Element&);
#endif

private:
    void adjustDisplayContentsStyle(RenderStyle&) const;
    void adjustForSiteSpecificQuirks(RenderStyle&) const;

    void adjustThemeStyle(RenderStyle&, const RenderStyle* userAgentAppearanceStyle) const;

    static OptionSet<EventListenerRegionType> computeEventListenerRegionTypes(const Document&, const RenderStyle&, const EventTarget&, OptionSet<EventListenerRegionType>);

    CheckedRef<const Document> m_document;
    const RenderStyle& m_parentStyle;
    const RenderStyle& m_parentBoxStyle;
    RefPtr<Element> m_element;
};

}
}
