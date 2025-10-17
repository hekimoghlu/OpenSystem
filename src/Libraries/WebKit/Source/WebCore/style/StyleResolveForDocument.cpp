/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 17, 2024.
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
#include "StyleResolveForDocument.h"

#include "CSSFontSelector.h"
#include "Document.h"
#include "DocumentInlines.h"
#include "FontCascade.h"
#include "HTMLIFrameElement.h"
#include "LocalFrame.h"
#include "LocalFrameView.h"
#include "LocaleToScriptMapping.h"
#include "NodeRenderStyle.h"
#include "Page.h"
#include "RenderObject.h"
#include "RenderStyleSetters.h"
#include "RenderView.h"
#include "Settings.h"
#include "StyleAdjuster.h"
#include "StyleFontSizeFunctions.h"
#include "StyleResolver.h"

namespace WebCore {

namespace Style {

RenderStyle resolveForDocument(const Document& document)
{
    ASSERT(document.hasLivingRenderTree());

    RenderView& renderView = *document.renderView();

    auto documentStyle = RenderStyle::create();

    documentStyle.setDisplay(DisplayType::Block);
    documentStyle.setRTLOrdering(document.visuallyOrdered() ? Order::Visual : Order::Logical);
    documentStyle.setZoom(!document.printing() ? renderView.frame().pageZoomFactor() : 1);
    documentStyle.setPageScaleTransform(renderView.frame().frameScaleFactor());

    // This overrides any -webkit-user-modify inherited from the parent iframe.
    documentStyle.setUserModify(document.inDesignMode() ? UserModify::ReadWrite : UserModify::ReadOnly);
#if PLATFORM(IOS_FAMILY)
    if (document.inDesignMode())
        documentStyle.setTextSizeAdjust(TextSizeAdjustment::none());
#endif

    Adjuster::adjustEventListenerRegionTypesForRootStyle(documentStyle, document);
    
    const Pagination& pagination = renderView.frameView().pagination();
    if (pagination.mode != Pagination::Mode::Unpaginated) {
        documentStyle.setColumnStylesFromPaginationMode(pagination.mode);
        documentStyle.setColumnGap(GapLength(WebCore::Length(static_cast<int>(pagination.gap), LengthType::Fixed)));
        if (renderView.multiColumnFlow())
            renderView.updateColumnProgressionFromStyle(documentStyle);
    }

    auto fontDescription = [&]() {
        auto& settings = renderView.frame().settings();

        FontCascadeDescription fontDescription;
        fontDescription.setSpecifiedLocale(document.contentLanguage());
        fontDescription.setOneFamily(standardFamily);
        fontDescription.setShouldAllowUserInstalledFonts(settings.shouldAllowUserInstalledFonts() ? AllowUserInstalledFonts::Yes : AllowUserInstalledFonts::No);

        fontDescription.setKeywordSizeFromIdentifier(CSSValueMedium);
        int size = fontSizeForKeyword(CSSValueMedium, false, document);
        fontDescription.setSpecifiedSize(size);
        bool useSVGZoomRules = document.isSVGDocument();
        fontDescription.setComputedSize(computedFontSizeFromSpecifiedSize(size, fontDescription.isAbsoluteSize(), useSVGZoomRules, &documentStyle, document));

        auto [fontOrientation, glyphOrientation] = documentStyle.fontAndGlyphOrientation();
        fontDescription.setOrientation(fontOrientation);
        fontDescription.setNonCJKGlyphOrientation(glyphOrientation);
        return fontDescription;
    }();

    auto fontCascade = FontCascade { WTFMove(fontDescription), documentStyle.fontCascade() };

    // We don't just call setFontDescription() because we need to provide the fontSelector to the FontCascade.
    RefPtr fontSelector = document.protectedFontSelector();
    fontCascade.update(WTFMove(fontSelector));
    documentStyle.setFontCascade(WTFMove(fontCascade));

    return documentStyle;
}

}
}
