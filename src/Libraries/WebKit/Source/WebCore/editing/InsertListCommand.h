/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 31, 2023.
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

class HTMLElement;
class HTMLQualifiedName;

class InsertListCommand final : public CompositeEditCommand {
public:
    enum class Type : uint8_t { OrderedList, UnorderedList };

    static Ref<InsertListCommand> create(Ref<Document>&& document, Type listType)
    {
        return adoptRef(*new InsertListCommand(WTFMove(document), listType));
    }

    static RefPtr<HTMLElement> insertList(Ref<Document>&&, Type);
    
    bool preservesTypingStyle() const final { return true; }

private:
    InsertListCommand(Ref<Document>&&, Type);

    void doApply() final;
    EditAction editingAction() const final;

    HTMLElement* fixOrphanedListChild(Node&);
    bool selectionHasListOfType(const VisibleSelection&, const HTMLQualifiedName&);
    Ref<HTMLElement> mergeWithNeighboringLists(HTMLElement&);
    void doApplyForSingleParagraph(bool forceCreateList, const HTMLQualifiedName&, SimpleRange& currentSelection);
    void unlistifyParagraph(const VisiblePosition& originalStart, HTMLElement& listNode, Node* listChildNode);
    RefPtr<HTMLElement> listifyParagraph(const VisiblePosition& originalStart, const HTMLQualifiedName& listTag);
    RefPtr<HTMLElement> m_listElement;
    Type m_type;
};

} // namespace WebCore
