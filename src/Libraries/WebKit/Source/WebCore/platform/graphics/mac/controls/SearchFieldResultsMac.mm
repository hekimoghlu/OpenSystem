/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 1, 2024.
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
#import "config.h"
#import "SearchFieldResultsMac.h"

#if PLATFORM(MAC)

#import "ControlFactoryMac.h"
#import "FloatRoundedRect.h"
#import "GraphicsContext.h"
#import "LocalDefaultSystemAppearance.h"
#import "SearchFieldResultsPart.h"

namespace WebCore {

SearchFieldResultsMac::SearchFieldResultsMac(SearchFieldResultsPart& owningPart, ControlFactoryMac& controlFactory, NSSearchFieldCell *searchFieldCell, NSMenu *searchMenuTemplate)
    : SearchControlMac(owningPart, controlFactory, searchFieldCell)
    , m_searchMenuTemplate(searchMenuTemplate)
{
    ASSERT(owningPart.type() == StyleAppearance::SearchFieldResultsButton || owningPart.type() == StyleAppearance::SearchFieldResultsDecoration);
    ASSERT(searchFieldCell);
    ASSERT((owningPart.type() == StyleAppearance::SearchFieldResultsButton) == (searchMenuTemplate != nullptr));
}

void SearchFieldResultsMac::updateCellStates(const FloatRect& rect, const ControlStyle& style)
{
    SearchControlMac::updateCellStates(rect, style);

    if ([m_searchFieldCell searchMenuTemplate] != m_searchMenuTemplate)
        [m_searchFieldCell setSearchMenuTemplate:m_searchMenuTemplate.get()];
}

void SearchFieldResultsMac::draw(GraphicsContext& context, const FloatRoundedRect& borderRect, float deviceScaleFactor, const ControlStyle& style)
{
    LocalDefaultSystemAppearance localAppearance(style.states.contains(ControlStyle::State::DarkAppearance), style.accentColor);

    GraphicsContextStateSaver stateSaver(context);

    auto logicalRect = borderRect.rect();

    if (style.zoomFactor != 1) {
        logicalRect.scale(1 / style.zoomFactor);
        context.scale(style.zoomFactor);
    }

    drawCell(context, logicalRect, deviceScaleFactor, style, [m_searchFieldCell searchButtonCell], true);
}

} // namespace WebCore

#endif // PLATFORM(MAC)
