/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 18, 2024.
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
#import "SearchControlMac.h"
#import <wtf/TZoneMallocInlines.h>

#if PLATFORM(MAC)

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SearchControlMac);

SearchControlMac::SearchControlMac(ControlPart& part, ControlFactoryMac& controlFactory, NSSearchFieldCell *searchFieldCell)
    : ControlMac(part, controlFactory)
    , m_searchFieldCell(searchFieldCell)
{
    ASSERT(m_searchFieldCell);
}

void SearchControlMac::updateCellStates(const FloatRect& rect, const ControlStyle& style)
{
    ControlMac::updateCellStates(rect, style);

    [m_searchFieldCell setPlaceholderString:@""];
    [m_searchFieldCell setControlSize:controlSizeForFont(style)];

    // Update the various states we respond to.
    updateEnabledState(m_searchFieldCell.get(), style);
    updateFocusedState(m_searchFieldCell.get(), style);
}

} // namespace WebCore

#endif // PLATFORM(MAC)
