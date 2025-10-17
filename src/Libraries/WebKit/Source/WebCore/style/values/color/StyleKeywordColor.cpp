/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 15, 2022.
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
#include "StyleKeywordColor.h"

#include "CSSKeywordColor.h"
#include "Document.h"
#include "RenderStyle.h"
#include "RenderTheme.h"
#include "StyleBuilderState.h"
#include "StyleColor.h"
#include "StyleColorResolutionState.h"
#include "StyleCurrentColor.h"

namespace WebCore {
namespace Style {

Color toStyleColor(const CSS::KeywordColor& unresolved, ColorResolutionState& state)
{
    Ref protectedDocument = state.document;

    switch (unresolved.valueID) {
    case CSSValueInternalDocumentTextColor:
        return { protectedDocument->textColor() };
    case CSSValueWebkitLink:
        return { state.forVisitedLink == ForVisitedLink::Yes ? protectedDocument->visitedLinkColor(state.style) : protectedDocument->linkColor(state.style) };
    case CSSValueWebkitActivelink:
        return { protectedDocument->activeLinkColor(state.style) };
    case CSSValueWebkitFocusRingColor:
        return { RenderTheme::singleton().focusRingColor(protectedDocument->styleColorOptions(&state.style)) };
    case CSSValueCurrentcolor:
        return { CurrentColor() };
    default:
        return { CSS::colorFromKeyword(unresolved.valueID, protectedDocument->styleColorOptions(&state.style)) };
    }
}

} // namespace Style
} // namespace WebCore
