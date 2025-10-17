/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 3, 2022.
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
#include "EditingRange.h"

#include <WebCore/FrameSelection.h>
#include <WebCore/LocalFrame.h>
#include <WebCore/TextIterator.h>
#include <WebCore/VisibleUnits.h>

namespace WebKit {

std::optional<WebCore::SimpleRange> EditingRange::toRange(WebCore::LocalFrame& frame, const EditingRange& editingRange, EditingRangeIsRelativeTo base)
{
    WebCore::CharacterRange range { editingRange.location, editingRange.length };

    if (base == EditingRangeIsRelativeTo::EditableRoot) {
        // Our critical assumption is that this code path is called by input methods that
        // concentrate on a given area containing the selection.
        // We have to do this because of text fields and textareas. The DOM for those is not
        // directly in the document DOM, so serialization is problematic. Our solution is
        // to use the root editable element of the selection start as the positional base.
        // That fits with AppKit's idea of an input context.
        RefPtr element = frame.selection().rootEditableElementOrDocumentElement();
        if (!element)
            return std::nullopt;
        return resolveCharacterRange(makeRangeSelectingNodeContents(*element), range);
    }

    ASSERT(base == EditingRangeIsRelativeTo::Paragraph);

    auto paragraphStart = makeBoundaryPoint(startOfParagraph(frame.selection().selection().visibleStart()));
    if (!paragraphStart)
        return std::nullopt;

    auto scopeEnd = makeBoundaryPointAfterNodeContents(Ref { paragraphStart->container->treeScope().rootNode() });
    return WebCore::resolveCharacterRange({ WTFMove(*paragraphStart), WTFMove(scopeEnd) }, range);
}

EditingRange EditingRange::fromRange(WebCore::LocalFrame& frame, const std::optional<WebCore::SimpleRange>& range, EditingRangeIsRelativeTo editingRangeIsRelativeTo)
{
    ASSERT(editingRangeIsRelativeTo == EditingRangeIsRelativeTo::EditableRoot);

    if (!range)
        return { };

    RefPtr element = frame.selection().rootEditableElementOrDocumentElement();
    if (!element)
        return { };

    auto relativeRange = characterRange(makeBoundaryPointBeforeNodeContents(*element), *range);
    return EditingRange(relativeRange.location, relativeRange.length);
}

} // namespace WebKit
