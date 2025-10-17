/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 21, 2023.
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

// ModifySelectionListLevelCommand provides functions useful for both increasing and decreasing the list level.
// It is the base class of IncreaseSelectionListLevelCommand and DecreaseSelectionListLevelCommand.
// It is not used on its own.
class ModifySelectionListLevelCommand : public CompositeEditCommand {
protected:
    explicit ModifySelectionListLevelCommand(Ref<Document>&&);
    
    void appendSiblingNodeRange(Node* startNode, Node* endNode, Element* newParent);
    void insertSiblingNodeRangeBefore(Node* startNode, Node* endNode, Node* refNode);
    void insertSiblingNodeRangeAfter(Node* startNode, Node* endNode, Node* refNode);

private:
    bool preservesTypingStyle() const override;
};

// IncreaseSelectionListLevelCommand moves the selected list items one level deeper.
class IncreaseSelectionListLevelCommand : public ModifySelectionListLevelCommand {
public:
    enum class Type : uint8_t { InheritedListType, OrderedList, UnorderedList };
    static Ref<IncreaseSelectionListLevelCommand> create(Document& document, Type type)
    {
        return adoptRef(*new IncreaseSelectionListLevelCommand(document, type));
    }

    static bool canIncreaseSelectionListLevel(Document*);
    static RefPtr<Node> increaseSelectionListLevel(Document*);
    static RefPtr<Node> increaseSelectionListLevelOrdered(Document*);
    static RefPtr<Node> increaseSelectionListLevelUnordered(Document*);

private:
    static RefPtr<Node> increaseSelectionListLevel(Document*, Type);
    
    IncreaseSelectionListLevelCommand(Ref<Document>&&, Type);

    void doApply() override;

    Type m_listType;
    RefPtr<Node> m_listElement;
};

// DecreaseSelectionListLevelCommand moves the selected list items one level shallower.
class DecreaseSelectionListLevelCommand : public ModifySelectionListLevelCommand {
public:
    static bool canDecreaseSelectionListLevel(Document*);
    static void decreaseSelectionListLevel(Document*);

private:
    static Ref<DecreaseSelectionListLevelCommand> create(Ref<Document>&& document)
    {
        return adoptRef(*new DecreaseSelectionListLevelCommand(WTFMove(document)));
    }

    explicit DecreaseSelectionListLevelCommand(Ref<Document>&&);

    void doApply() override;
};

} // namespace WebCore
