/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 27, 2024.
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
#include "ChangeListTypeCommand.h"

#include "Editing.h"
#include "ElementAncestorIteratorInlines.h"
#include "FrameDestructionObserverInlines.h"
#include "FrameSelection.h"
#include "HTMLElement.h"
#include "HTMLOListElement.h"
#include "HTMLUListElement.h"
#include "LocalFrame.h"
#include <wtf/RefPtr.h>

namespace WebCore {

static std::optional<std::pair<ChangeListTypeCommand::Type, Ref<HTMLElement>>> listConversionTypeForSelection(const VisibleSelection& selection)
{
    auto startNode = selection.start().containerNode();
    auto endNode = selection.end().containerNode();
    if (!startNode || !endNode)
        return { };
    auto commonAncestor = commonInclusiveAncestor<ComposedTree>(*startNode, *endNode);

    RefPtr<HTMLElement> listToReplace;
    if (auto* htmlElement = dynamicDowncast<HTMLElement>(commonAncestor); is<HTMLUListElement>(htmlElement) || is<HTMLOListElement>(htmlElement))
        listToReplace = htmlElement;
    else
        listToReplace = enclosingList(commonAncestor);

    if (is<HTMLUListElement>(listToReplace))
        return {{ ChangeListTypeCommand::Type::ConvertToOrderedList, listToReplace.releaseNonNull() }};

    if (is<HTMLOListElement>(listToReplace))
        return {{ ChangeListTypeCommand::Type::ConvertToUnorderedList, listToReplace.releaseNonNull() }};

    return std::nullopt;
}

std::optional<ChangeListTypeCommand::Type> ChangeListTypeCommand::listConversionType(Document& document)
{
    if (RefPtr frame = document.frame()) {
        if (auto typeAndElement = listConversionTypeForSelection(frame->selection().selection()))
            return typeAndElement->first;
    }
    return std::nullopt;
}

Ref<HTMLElement> ChangeListTypeCommand::createNewList(const HTMLElement& listToReplace)
{
    RefPtr<HTMLElement> list;
    if (m_type == Type::ConvertToOrderedList)
        list = HTMLOListElement::create(document());
    else
        list = HTMLUListElement::create(document());
    list->cloneDataFromElement(listToReplace);
    return list.releaseNonNull();
}

void ChangeListTypeCommand::doApply()
{
    auto typeAndElement = listConversionTypeForSelection(endingSelection());
    if (!typeAndElement || typeAndElement->first != m_type)
        return;

    auto listToReplace = WTFMove(typeAndElement->second);
    auto newList = createNewList(listToReplace);
    insertNodeBefore(newList.copyRef(), listToReplace);
    moveRemainingSiblingsToNewParent(listToReplace->firstChild(), nullptr, newList);
    removeNode(listToReplace);
    setEndingSelection({ Position { newList.ptr(), Position::PositionIsAfterChildren }});
}

} // namespace WebCore
