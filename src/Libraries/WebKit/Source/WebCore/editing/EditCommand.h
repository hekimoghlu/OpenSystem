/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 22, 2022.
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

#include "AXTextStateChangeIntent.h"
#include "EditAction.h"
#include "VisibleSelection.h"
#include <wtf/WeakPtr.h>

#ifndef NDEBUG
#include <wtf/HashSet.h>
#endif

namespace WebCore {

class CompositeEditCommand;
class Document;
class Element;

ASCIILiteral inputTypeNameForEditingAction(EditAction);
bool isInputMethodComposingForEditingAction(EditAction);

using NodeSet = UncheckedKeyHashSet<Ref<Node>>;

class EditCommand : public RefCounted<EditCommand> {
public:
    virtual ~EditCommand();

    void setParent(RefPtr<CompositeEditCommand>&&);

    virtual EditAction editingAction() const;

    const VisibleSelection& startingSelection() const { return m_startingSelection; }
    const VisibleSelection& endingSelection() const { return m_endingSelection; }

    virtual bool isInsertTextCommand() const { return false; }    
    virtual bool isSimpleEditCommand() const { return false; }
    virtual bool isCompositeEditCommand() const { return false; }
    bool isTopLevelCommand() const { return !m_parent; }

    virtual void doApply() = 0;

protected:
    explicit EditCommand(Ref<Document>&&, EditAction = EditAction::Unspecified);
    EditCommand(Ref<Document>&&, const VisibleSelection&, const VisibleSelection&);

    Ref<Document> protectedDocument() const { return m_document.copyRef(); }
    const Document& document() const { return m_document; }
    Document& document() { return m_document; }
    CompositeEditCommand* parent() const { return m_parent.get(); }
    void setStartingSelection(const VisibleSelection&);
    WEBCORE_EXPORT void setEndingSelection(const VisibleSelection&);

    bool isEditingTextAreaOrTextInput() const;

    void postTextStateChangeNotification(AXTextEditType, const String&);
    void postTextStateChangeNotification(AXTextEditType, const String&, const VisiblePosition&);

private:
    Ref<Document> m_document;
    VisibleSelection m_startingSelection;
    VisibleSelection m_endingSelection;
    WeakPtr<CompositeEditCommand> m_parent;
    EditAction m_editingAction { EditAction::Unspecified };
};

enum ShouldAssumeContentIsAlwaysEditable {
    AssumeContentIsAlwaysEditable,
    DoNotAssumeContentIsAlwaysEditable,
};

class SimpleEditCommand : public EditCommand {
public:
    virtual void doUnapply() = 0;
    virtual void doReapply(); // calls doApply()

#ifndef NDEBUG
    virtual void getNodesInCommand(NodeSet&) = 0;
#endif

protected:
    explicit SimpleEditCommand(Ref<Document>&&, EditAction = EditAction::Unspecified);

#ifndef NDEBUG
    void addNodeAndDescendants(Node*, NodeSet&);
#endif

private:
    bool isSimpleEditCommand() const override { return true; }
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::SimpleEditCommand)
    static bool isType(const WebCore::EditCommand& command) { return command.isSimpleEditCommand(); }
SPECIALIZE_TYPE_TRAITS_END()
