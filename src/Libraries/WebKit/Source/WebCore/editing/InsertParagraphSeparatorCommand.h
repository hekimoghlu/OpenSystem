/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 26, 2024.
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

class InsertParagraphSeparatorCommand : public CompositeEditCommand {
public:
    static Ref<InsertParagraphSeparatorCommand> create(Ref<Document>&& document, bool useDefaultParagraphElement = false, bool pasteBlockqutoeIntoUnquotedArea = false, EditAction editingAction = EditAction::Insert)
    {
        return adoptRef(*new InsertParagraphSeparatorCommand(WTFMove(document), useDefaultParagraphElement, pasteBlockqutoeIntoUnquotedArea, editingAction));
    }

private:
    InsertParagraphSeparatorCommand(Ref<Document>&&, bool useDefaultParagraphElement, bool pasteBlockqutoeIntoUnquotedArea, EditAction);

    void doApply() override;

    void calculateStyleBeforeInsertion(const Position&);
    void applyStyleAfterInsertion(Node* originalEnclosingBlock);
    void getAncestorsInsideBlock(const Node* insertionNode, Element* outerBlock, Vector<RefPtr<Element>>& ancestors);
    Ref<Element> cloneHierarchyUnderNewBlock(const Vector<RefPtr<Element>>& ancestors, Ref<Element>&& blockToInsert);

    bool shouldUseDefaultParagraphElement(Node*) const;

    bool preservesTypingStyle() const override;

    RefPtr<EditingStyle> protectedStyle() const { return m_style; }

    RefPtr<EditingStyle> m_style;

    bool m_mustUseDefaultParagraphElement;
    bool m_pasteBlockqutoeIntoUnquotedArea;
};

} // namespace WebCore
