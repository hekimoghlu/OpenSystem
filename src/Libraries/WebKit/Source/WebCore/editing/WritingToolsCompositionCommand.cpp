/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 28, 2022.
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
#include "WritingToolsCompositionCommand.h"

#include "FrameSelection.h"
#include "ReplaceSelectionCommand.h"
#include "TextIterator.h"

namespace WebCore {

WritingToolsCompositionCommand::WritingToolsCompositionCommand(Ref<Document>&& document, const SimpleRange& endingContextRange)
    : CompositeEditCommand(WTFMove(document), EditAction::InsertReplacement)
    , m_endingContextRange(endingContextRange)
    , m_currentContextRange(endingContextRange)
{
}

void WritingToolsCompositionCommand::replaceContentsOfRangeWithFragment(RefPtr<DocumentFragment>&& fragment, const SimpleRange& range, MatchStyle matchStyle, State state)
{
    auto contextRange = m_endingContextRange;

    auto contextRangeCount = characterCount(contextRange);
    auto resolvedCharacterRange = characterRange(contextRange, range);

    OptionSet<ReplaceSelectionCommand::CommandOption> options { ReplaceSelectionCommand::PreventNesting, ReplaceSelectionCommand::SanitizeFragment, ReplaceSelectionCommand::SelectReplacement };
    if (matchStyle == MatchStyle::Yes)
        options.add(ReplaceSelectionCommand::MatchStyle);

    applyCommandToComposite(ReplaceSelectionCommand::create(protectedDocument(), WTFMove(fragment), options, EditAction::InsertReplacement), range);

    // Restore the context range to what it previously was, while taking into account the newly replaced contents.
    auto newContextRange = rangeExpandedAroundRangeByCharacters(endingSelection(), resolvedCharacterRange.location, contextRangeCount - (resolvedCharacterRange.location + resolvedCharacterRange.length));
    if (!newContextRange) {
        ASSERT_NOT_REACHED();
        return;
    }

    m_currentContextRange = *newContextRange;

    if (state == State::Complete) {
        // When the command is signaled to be "complete", this commits the entire command as a whole to the undo/redo stack.
        commit();
    }
}

void WritingToolsCompositionCommand::commit()
{
    this->apply();
    m_endingContextRange = m_currentContextRange;
}

} // namespace WebCore
