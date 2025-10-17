/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 10, 2022.
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
#include "MoveSelectionCommand.h"

#include "DeleteSelectionCommand.h"
#include "DocumentFragment.h"
#include "ReplaceSelectionCommand.h"

namespace WebCore {

MoveSelectionCommand::MoveSelectionCommand(Ref<DocumentFragment>&& fragment, const Position& position, bool smartInsert, bool smartDelete)
    : CompositeEditCommand(position.anchorNode()->document())
    , m_fragment(WTFMove(fragment))
    , m_position(position)
    , m_smartInsert(smartInsert)
    , m_smartDelete(smartDelete)
{
}

void MoveSelectionCommand::doApply()
{
    ASSERT(endingSelection().isNonOrphanedRange());

    Position pos = m_position;
    if (pos.isNull())
        return;

    // Update the position otherwise it may become invalid after the selection is deleted.
    Position selectionEnd = endingSelection().end();
    if (pos.anchorType() == Position::PositionIsOffsetInAnchor && selectionEnd.anchorType() == Position::PositionIsOffsetInAnchor
        && selectionEnd.containerNode() == pos.containerNode() && selectionEnd.offsetInContainerNode() < pos.offsetInContainerNode()) {
        pos.moveToOffset(pos.offsetInContainerNode() - selectionEnd.offsetInContainerNode());

        Position selectionStart = endingSelection().start();
        if (selectionStart.anchorType() == Position::PositionIsOffsetInAnchor && selectionStart.containerNode() == pos.containerNode())
            pos.moveToOffset(pos.offsetInContainerNode() + selectionStart.offsetInContainerNode());
    }

    {
        auto deleteSelection = DeleteSelectionCommand::create(document(), m_smartDelete, true, false, true, true, EditAction::DeleteByDrag);
        deleteSelection->setParent(this);
        deleteSelection->apply();
        m_commands.append(WTFMove(deleteSelection));
    }

    // If the node for the destination has been removed as a result of the deletion,
    // set the destination to the ending point after the deletion.
    // Fixes: <rdar://problem/3910425> REGRESSION (Mail): Crash in ReplaceSelectionCommand; 
    //        selection is empty, leading to null deref
    if (!pos.anchorNode()->isConnected())
        pos = endingSelection().start();

    cleanupAfterDeletion(pos);

    setEndingSelection(VisibleSelection(pos, endingSelection().affinity(), endingSelection().directionality()));
    setStartingSelection(endingSelection());
    if (!pos.anchorNode()->isConnected()) {
        // Document was modified out from under us.
        return;
    }
    OptionSet<ReplaceSelectionCommand::CommandOption> options { ReplaceSelectionCommand::SelectReplacement, ReplaceSelectionCommand::PreventNesting };
    if (m_smartInsert)
        options.add(ReplaceSelectionCommand::SmartReplace);

    {
        auto replaceSelection = ReplaceSelectionCommand::create(document(), m_fragment.copyRef(), options, EditAction::InsertFromDrop);
        replaceSelection->setParent(this);
        replaceSelection->apply();
        m_commands.append(WTFMove(replaceSelection));
    }
}

EditAction MoveSelectionCommand::editingAction() const
{
    return EditAction::DeleteByDrag;
}

} // namespace WebCore
