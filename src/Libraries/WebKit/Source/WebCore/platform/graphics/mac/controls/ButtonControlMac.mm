/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 24, 2024.
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
#import "ButtonControlMac.h"

#if PLATFORM(MAC)

#import <pal/spi/cocoa/NSButtonCellSPI.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ButtonControlMac);

ButtonControlMac::ButtonControlMac(ControlPart& part, ControlFactoryMac& controlFactory, NSButtonCell *buttonCell)
    : ControlMac(part, controlFactory)
    , m_buttonCell(buttonCell)
{
    ASSERT(m_buttonCell);
}

void ButtonControlMac::updateCellStates(const FloatRect& rect, const ControlStyle& style)
{
    ControlMac::updateCellStates(rect, style);

    const auto& states = style.states;

    updatePressedState(m_buttonCell.get(), style);
    updateEnabledState(m_buttonCell.get(), style);
    updateCheckedState(m_buttonCell.get(), style);
    
    if (states.contains(ControlStyle::State::Presenting))
        [m_buttonCell _setHighlighted:YES animated:NO];

    // Window inactive state does not need to be checked explicitly, since we paint parented to
    // a view in a window whose key state can be detected.

    // Only update if we have to, since AppKit does work even if the size is the same.
    auto controlSize = controlSizeForSize(rect.size(), style);
    if (controlSize != [m_buttonCell controlSize])
        [m_buttonCell setControlSize:controlSize];
}

} // namespace WebCore

#endif // PLATFORM(MAC)
