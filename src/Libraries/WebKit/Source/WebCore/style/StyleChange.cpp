/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 15, 2023.
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
#include "StyleChange.h"

#include "RenderStyleConstants.h"
#include "RenderStyleInlines.h"
#include <wtf/text/AtomString.h>

namespace WebCore {
namespace Style {

Change determineChange(const RenderStyle& s1, const RenderStyle& s2)
{
    if (s1.display() != s2.display())
        return Change::Renderer;
    if (s1.hasPseudoStyle(PseudoId::FirstLetter) != s2.hasPseudoStyle(PseudoId::FirstLetter))
        return Change::Renderer;
    // We just detach if a renderer acquires or loses a column-span, since spanning elements
    // typically won't contain much content.
    auto columnSpanNeedsNewRenderer = [&] {
        if (s1.columnSpan() != s2.columnSpan())
            return true;
        if (s1.columnSpan() != ColumnSpan::All)
            return false;
        // Spanning in ignored for floating and out-of-flow boxes.
        return s1.isFloating() != s2.isFloating() || s1.hasOutOfFlowPosition() != s2.hasOutOfFlowPosition();
    }();
    if (columnSpanNeedsNewRenderer)
        return Change::Renderer;
    if (!s1.contentDataEquivalent(&s2))
        return Change::Renderer;
    // When text-combine property has been changed, we need to prepare a separate renderer object.
    // When text-combine is on, we use RenderCombineText, otherwise RenderText.
    // https://bugs.webkit.org/show_bug.cgi?id=55069
    if (s1.hasTextCombine() != s2.hasTextCombine())
        return Change::Renderer;

    // Query container changes affect descendant style.
    if (s1.containerType() != s2.containerType() || s1.containerNames() != s2.containerNames())
        return Change::Descendants;

    if (!s1.descendantAffectingNonInheritedPropertiesEqual(s2))
        return Change::Inherited;

    if (!s1.nonFastPathInheritedEqual(s2))
        return Change::Inherited;

    bool nonInheritedEqual = s1.nonInheritedEqual(s2);
    if (!s1.fastPathInheritedEqual(s2))
        return nonInheritedEqual ? Change::FastPathInherited : Change::NonInheritedAndFastPathInherited;

    if (!nonInheritedEqual)
        return Change::NonInherited;

    return Change::None;
}

}
}
