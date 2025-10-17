/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 21, 2022.
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
#include "DocumentFragment.h"
#include "NodeTraversal.h"

namespace WebCore {

class Range;
class ReplacementFragment;

class ReplaceSelectionCommand : public CompositeEditCommand {
public:
    enum CommandOption {
        SelectReplacement = 1 << 0,
        SmartReplace = 1 << 1,
        MatchStyle = 1 << 2,
        PreventNesting = 1 << 3,
        MovingParagraph = 1 << 4,
        SanitizeFragment = 1 << 5,
        IgnoreMailBlockquote = 1 << 6,
    };

    static Ref<ReplaceSelectionCommand> create(Ref<Document>&& document, RefPtr<DocumentFragment>&& fragment, OptionSet<CommandOption> options, EditAction editingAction = EditAction::Insert)
    {
        return adoptRef(*new ReplaceSelectionCommand(WTFMove(document), WTFMove(fragment), options, editingAction));
    }

    virtual ~ReplaceSelectionCommand();

    VisibleSelection visibleSelectionForInsertedText() const { return m_visibleSelectionForInsertedText; }
    String documentFragmentPlainText() const { return m_documentFragmentPlainText; }

    std::optional<SimpleRange> insertedContentRange() const;

private:
    ReplaceSelectionCommand(Ref<Document>&&, RefPtr<DocumentFragment>&&, OptionSet<CommandOption>, EditAction);

    String inputEventData() const final;
    RefPtr<DataTransfer> inputEventDataTransfer() const final;
    bool willApplyCommand() final;
    void doApply() override;

    class InsertedNodes {
    public:
        void respondToNodeInsertion(Node*);
        void willRemoveNodePreservingChildren(Node*);
        void willRemovePossibleAncestorNode(Node*);
        void willRemoveNode(Node*);
        void didReplaceNode(Node*, Node* newNode);

        bool isEmpty() { return !m_firstNodeInserted; }
        Node* firstNodeInserted() const { return m_firstNodeInserted.get(); }
        RefPtr<Node> protectedFirstNodeInserted() const { return m_firstNodeInserted; }
        Node* lastLeafInserted() const
        {
            ASSERT(m_lastNodeInserted);
            return m_lastNodeInserted->lastDescendant();
        }
        RefPtr<Node> protectedLastLeafInserted() const { return lastLeafInserted(); }
        Node* pastLastLeaf() const
        {
            ASSERT(m_lastNodeInserted);
            return NodeTraversal::next(*m_lastNodeInserted->lastDescendant());
        }

    private:
        RefPtr<Node> m_firstNodeInserted;
        RefPtr<Node> m_lastNodeInserted;
    };

    RefPtr<Node> insertAsListItems(HTMLElement& listElement, Node* insertionNode, const Position&, InsertedNodes&);

    void updateNodesInserted(Node*);
    bool shouldRemoveEndBR(Node*, const VisiblePosition&);
    
    bool shouldMergeStart(bool, bool, bool);
    bool shouldMergeEnd(bool selectionEndWasEndOfParagraph);
    bool shouldMerge(const VisiblePosition&, const VisiblePosition&);
    
    void mergeEndIfNeeded();
    
    void removeUnrenderedTextNodesAtEnds(InsertedNodes&);
    
    void removeRedundantStylesAndKeepStyleSpanInline(InsertedNodes&);
    void inverseTransformColor(InsertedNodes&);
    void makeInsertedContentRoundTrippableWithHTMLTreeBuilder(InsertedNodes&);
    void moveNodeOutOfAncestor(Node&, Node& ancestor, InsertedNodes&);
    void handleStyleSpans(InsertedNodes&);
    
    VisiblePosition positionAtStartOfInsertedContent() const;
    VisiblePosition positionAtEndOfInsertedContent() const;

    bool shouldPerformSmartReplace() const;
    bool shouldPerformSmartParagraphReplace() const;
    void addSpacesForSmartReplace();
    void addNewLinesForSmartReplace();
    void completeHTMLReplacement(const Position& lastPositionToSelect);
    void mergeTextNodesAroundPosition(Position&, Position& positionOnlyToBeUpdated);

    ReplacementFragment* ensureReplacementFragment();
    bool performTrivialReplace(const ReplacementFragment&);

    void updateDirectionForStartOfInsertedContentIfNeeded();

    RefPtr<DocumentFragment> protectedDocumentFragment() const { return m_documentFragment; }

    VisibleSelection m_visibleSelectionForInsertedText;
    Position m_startOfInsertedContent;
    Position m_endOfInsertedContent;
    RefPtr<EditingStyle> m_insertionStyle;
    bool m_selectReplacement;
    bool m_smartReplace;
    bool m_matchStyle;
    RefPtr<DocumentFragment> m_documentFragment;
    std::unique_ptr<ReplacementFragment> m_replacementFragment;
    String m_documentFragmentHTMLMarkup;
    String m_documentFragmentPlainText;
    bool m_preventNesting;
    bool m_movingParagraph;
    bool m_sanitizeFragment;
    bool m_shouldMergeEnd;
    bool m_ignoreMailBlockquote;
};

} // namespace WebCore
