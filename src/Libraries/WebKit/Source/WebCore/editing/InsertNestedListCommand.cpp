/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 25, 2024.
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
#include "InsertNestedListCommand.h"

#include "Editing.h"
#include "HTMLLIElement.h"
#include "HTMLNames.h"
#include "InsertListCommand.h"
#include "ModifySelectionListLevel.h"

namespace WebCore {

void InsertNestedListCommand::insertUnorderedList(Document& document)
{
    InsertNestedListCommand::create(document, Type::UnorderedList)->apply();
}

void InsertNestedListCommand::insertOrderedList(Document& document)
{
    InsertNestedListCommand::create(document, Type::OrderedList)->apply();
}

void InsertNestedListCommand::doApply()
{
    if (endingSelection().isNoneOrOrphaned() || !endingSelection().isContentRichlyEditable())
        return;

    if (RefPtr enclosingItem = enclosingElementWithTag(endingSelection().visibleStart().deepEquivalent(), HTMLNames::liTag)) {
        auto newListItem = HTMLLIElement::create(document());
        insertNodeAfter(newListItem.copyRef(), *enclosingItem);
        setEndingSelection({ Position { newListItem.ptr(), Position::PositionIsBeforeChildren }, Affinity::Downstream });

        auto commandType = m_type == Type::OrderedList ? IncreaseSelectionListLevelCommand::Type::OrderedList : IncreaseSelectionListLevelCommand::Type::UnorderedList;
        applyCommandToComposite(IncreaseSelectionListLevelCommand::create(document(), commandType));
        return;
    }

    auto commandType = m_type == Type::OrderedList ? InsertListCommand::Type::OrderedList : InsertListCommand::Type::UnorderedList;
    applyCommandToComposite(InsertListCommand::create(document(), commandType));
}

} // namespace WebCore
