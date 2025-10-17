/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 21, 2022.
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

#include "CompositeEditCommand.h"

namespace WebCore {

class EditingStyle;

class DeleteSelectionCommand : public CompositeEditCommand { 
public:
    static Ref<DeleteSelectionCommand> create(Ref<Document>&& document, bool smartDelete = false, bool mergeBlocksAfterDelete = true, bool replace = false, bool expandForSpecialElements = false, bool sanitizeMarkup = true, EditAction editingAction = EditAction::Delete)
    {
        return adoptRef(*new DeleteSelectionCommand(WTFMove(document), smartDelete, mergeBlocksAfterDelete, replace, expandForSpecialElements, sanitizeMarkup, editingAction));
    }
    static Ref<DeleteSelectionCommand> create(const VisibleSelection& selection, bool smartDelete = false, bool mergeBlocksAfterDelete = true, bool replace = false, bool expandForSpecialElements = false, bool sanitizeMarkup = true, EditAction editingAction = EditAction::Delete)
    {
        return adoptRef(*new DeleteSelectionCommand(selection, smartDelete, mergeBlocksAfterDelete, replace, expandForSpecialElements, sanitizeMarkup, editingAction));
    }

protected:
    DeleteSelectionCommand(Ref<Document>&&, bool smartDelete, bool mergeBlocksAfterDelete, bool replace, bool expandForSpecialElements, bool santizeMarkup, EditAction = EditAction::Delete);

private:
    DeleteSelectionCommand(const VisibleSelection&, bool smartDelete, bool mergeBlocksAfterDelete, bool replace, bool expandForSpecialElements, bool sanitizeMarkup, EditAction);

    void doApply() override;
    
    bool preservesTypingStyle() const override;

    void initializeStartEnd(Position&, Position&);
    void setStartingSelectionOnSmartDelete(const Position&, const Position&);
    bool initializePositionData();
    void saveTypingStyleState();
    bool handleSpecialCaseBRDelete();
    void handleGeneralDelete();
    void fixupWhitespace();
    void mergeParagraphs();
    void removePreviouslySelectedEmptyTableRows();
    void calculateTypingStyleAfterDelete();
    void clearTransientState();
    void makeStylingElementsDirectChildrenOfEditableRootToPreventStyleLoss();
    void removeNode(Node&, ShouldAssumeContentIsAlwaysEditable = DoNotAssumeContentIsAlwaysEditable) override;
    void deleteTextFromNode(Text&, unsigned, unsigned) override;
    void removeRedundantBlocks();
    bool shouldSmartDeleteParagraphSpacers();
    void smartDeleteParagraphSpacers();

    // This function provides access to original string after the correction has been deleted.
    String originalStringForAutocorrectionAtBeginningOfSelection();

    void removeNodeUpdatingStates(Node&, ShouldAssumeContentIsAlwaysEditable);
    void insertBlockPlaceholderForTableCellIfNeeded(Element&);

    RefPtr<Node> protectedStartBlock() const { return m_startBlock; }
    RefPtr<Node> protectedEndBlock() const { return m_endBlock; }
    RefPtr<Node> protectedEndTableRow() const { return m_endTableRow; }

    bool m_hasSelectionToDelete;
    bool m_smartDelete;
    bool m_mergeBlocksAfterDelete;
    bool m_needPlaceholder;
    bool m_replace;
    bool m_expandForSpecialElements;
    bool m_pruneStartBlockIfNecessary;
    bool m_startsAtEmptyLine;
    bool m_sanitizeMarkup;

    // This data is transient and should be cleared at the end of the doApply function.
    VisibleSelection m_selectionToDelete;
    Position m_upstreamStart;
    Position m_downstreamStart;
    Position m_upstreamEnd;
    Position m_downstreamEnd;
    Position m_endingPosition;
    Position m_leadingWhitespace;
    Position m_trailingWhitespace;
    RefPtr<Node> m_startBlock;
    RefPtr<Node> m_endBlock;
    RefPtr<EditingStyle> m_typingStyle;
    RefPtr<EditingStyle> m_deleteIntoBlockquoteStyle;
    RefPtr<Node> m_startRoot;
    RefPtr<Node> m_endRoot;
    RefPtr<Node> m_startTableRow;
    RefPtr<Node> m_endTableRow;
    RefPtr<Node> m_temporaryPlaceholder;
};

} // namespace WebCore
