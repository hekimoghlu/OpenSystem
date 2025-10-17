/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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
#include "EditCommand.h"

#include "AXObjectCache.h"
#include "CompositeEditCommand.h"
#include "DocumentInlines.h"
#include "Editing.h"
#include "Editor.h"
#include "Element.h"
#include "HTMLTextFormControlElement.h"
#include "NodeTraversal.h"

namespace WebCore {

ASCIILiteral inputTypeNameForEditingAction(EditAction action)
{
    switch (action) {
    case EditAction::Justify:
        return "formatJustifyFull"_s;
    case EditAction::AlignLeft:
        return "formatJustifyLeft"_s;
    case EditAction::AlignRight:
        return "formatJustifyRight"_s;
    case EditAction::Center:
        return "formatJustifyCenter"_s;
    case EditAction::Subscript:
        return "formatSubscript"_s;
    case EditAction::Superscript:
        return "formatSuperscript"_s;
    case EditAction::Underline:
        return "formatUnderline"_s;
    case EditAction::StrikeThrough:
        return "formatStrikeThrough"_s;
    case EditAction::SetColor:
        return "formatFontColor"_s;
    case EditAction::DeleteByDrag:
        return "deleteByDrag"_s;
    case EditAction::Cut:
        return "deleteByCut"_s;
    case EditAction::Bold:
        return "formatBold"_s;
    case EditAction::Italics:
        return "formatItalic"_s;
    case EditAction::Paste:
        return "insertFromPaste"_s;
    case EditAction::Delete:
    case EditAction::TypingDeleteSelection:
        return "deleteContent"_s;
    case EditAction::TypingDeleteBackward:
        return "deleteContentBackward"_s;
    case EditAction::TypingDeleteForward:
        return "deleteContentForward"_s;
    case EditAction::TypingDeleteWordBackward:
        return "deleteWordBackward"_s;
    case EditAction::TypingDeleteWordForward:
        return "deleteWordForward"_s;
    case EditAction::TypingDeleteLineBackward:
        return "deleteHardLineBackward"_s;
    case EditAction::TypingDeleteLineForward:
        return "deleteHardLineForward"_s;
    case EditAction::TypingDeletePendingComposition:
        return "deleteCompositionText"_s;
    case EditAction::TypingDeleteFinalComposition:
        return "deleteByComposition"_s;
    case EditAction::Insert:
    case EditAction::TypingInsertText:
        return "insertText"_s;
    case EditAction::InsertReplacement:
        return "insertReplacementText"_s;
    case EditAction::InsertFromDrop:
        return "insertFromDrop"_s;
    case EditAction::TypingInsertLineBreak:
        return "insertLineBreak"_s;
    case EditAction::TypingInsertParagraph:
        return "insertParagraph"_s;
    case EditAction::InsertOrderedList:
        return "insertOrderedList"_s;
    case EditAction::InsertUnorderedList:
        return "insertUnorderedList"_s;
    case EditAction::TypingInsertPendingComposition:
        return "insertCompositionText"_s;
    case EditAction::TypingInsertFinalComposition:
        return "insertFromComposition"_s;
    case EditAction::Indent:
        return "formatIndent"_s;
    case EditAction::Outdent:
        return "formatOutdent"_s;
    case EditAction::SetInlineWritingDirection:
        return "formatSetInlineTextDirection"_s;
    case EditAction::SetBlockWritingDirection:
        return "formatSetBlockTextDirection"_s;
    case EditAction::CreateLink:
        return "insertLink"_s;
    default:
        return ""_s;
    }
}

bool isInputMethodComposingForEditingAction(EditAction action)
{
    switch (action) {
    case EditAction::TypingDeletePendingComposition:
    case EditAction::TypingDeleteFinalComposition:
    case EditAction::TypingInsertPendingComposition:
    case EditAction::TypingInsertFinalComposition:
        return true;
    default:
        break;
    }
    return false;
}

EditCommand::EditCommand(Ref<Document>&& document, EditAction editingAction)
    : m_document { WTFMove(document) }
    , m_startingSelection { m_document->selection().selection() }
    , m_endingSelection { m_startingSelection }
    , m_editingAction { editingAction }
{
}

EditCommand::EditCommand(Ref<Document>&& document, const VisibleSelection& startingSelection, const VisibleSelection& endingSelection)
    : m_document { WTFMove(document) }
    , m_startingSelection { startingSelection }
    , m_endingSelection { endingSelection }
{
}

EditCommand::~EditCommand() = default;

EditAction EditCommand::editingAction() const
{
    return m_editingAction;
}

static RefPtr<EditCommandComposition> compositionIfPossible(EditCommand& command)
{
    auto* compositeCommand = dynamicDowncast<CompositeEditCommand>(command);
    return compositeCommand ? compositeCommand->composition() : nullptr;
}

bool EditCommand::isEditingTextAreaOrTextInput() const
{
    return enclosingTextFormControl(m_document->selection().selection().start());
}

void EditCommand::setStartingSelection(const VisibleSelection& selection)
{
    for (RefPtr command = this; ; command = command->m_parent.get()) {
        if (auto composition = compositionIfPossible(*command))
            composition->setStartingSelection(selection);
        command->m_startingSelection = selection;
        if (!command->m_parent || command->m_parent->isFirstCommand(command.get()))
            break;
    }
}

void EditCommand::setEndingSelection(const VisibleSelection& selection)
{
    for (RefPtr command = this; command; command = command->m_parent.get()) {
        if (auto composition = compositionIfPossible(*command))
            composition->setEndingSelection(selection);
        command->m_endingSelection = selection;
    }
}

void EditCommand::setParent(RefPtr<CompositeEditCommand>&& parent)
{
    ASSERT((parent && !m_parent) || (!parent && m_parent));
    m_parent = WTFMove(parent);
    if (m_parent) {
        m_startingSelection = m_parent->m_endingSelection;
        m_endingSelection = m_parent->m_endingSelection;
    }
}

void EditCommand::postTextStateChangeNotification(AXTextEditType type, const String& text)
{
    if (!AXObjectCache::accessibilityEnabled())
        return;
    postTextStateChangeNotification(type, text, m_document->selection().selection().start());
}

void EditCommand::postTextStateChangeNotification(AXTextEditType type, const String& text, const VisiblePosition& position)
{
    if (!AXObjectCache::accessibilityEnabled())
        return;
    if (!text.length())
        return;
    auto document = protectedDocument();
    CheckedPtr cache = document->existingAXObjectCache();
    if (!cache)
        return;
    RefPtr node { highestEditableRoot(position.deepEquivalent(), HasEditableAXRole) };
    cache->postTextStateChangeNotification(node.get(), type, text, position);
}

SimpleEditCommand::SimpleEditCommand(Ref<Document>&& document, EditAction editingAction)
    : EditCommand(WTFMove(document), editingAction)
{
}

void SimpleEditCommand::doReapply()
{
    doApply();
}

#ifndef NDEBUG
void SimpleEditCommand::addNodeAndDescendants(Node* startNode, NodeSet& nodes)
{
    for (RefPtr node = startNode; node; node = NodeTraversal::next(*node, startNode))
        nodes.add(*node);
}
#endif

} // namespace WebCore
