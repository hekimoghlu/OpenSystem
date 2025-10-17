/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 13, 2025.
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
#include "TextInsertionBaseCommand.h"

#include "BeforeTextInsertedEvent.h"
#include "Document.h"
#include "Element.h"
#include "FrameSelection.h"
#include "LocalFrame.h"
#include "Node.h"

namespace WebCore {

TextInsertionBaseCommand::TextInsertionBaseCommand(Ref<Document>&& document, EditAction editingAction)
    : CompositeEditCommand(WTFMove(document), editingAction)
{
}

void TextInsertionBaseCommand::applyTextInsertionCommand(LocalFrame* frame, TextInsertionBaseCommand& command, const VisibleSelection& selectionForInsertion, const VisibleSelection& endingSelection)
{
    bool changeSelection = selectionForInsertion != endingSelection;
    if (changeSelection) {
        command.setStartingSelection(selectionForInsertion);
        command.setEndingSelection(selectionForInsertion);
    }
    command.apply();
    if (changeSelection) {
        command.setEndingSelection(endingSelection);
        frame->selection().setSelection(endingSelection);
    }
}

String dispatchBeforeTextInsertedEvent(const String& text, const VisibleSelection& selectionForInsertion, bool insertionIsForUpdatingComposition)
{
    if (insertionIsForUpdatingComposition)
        return text;

    String newText = text;
    if (RefPtr startNode = selectionForInsertion.start().containerNode()) {
        if (startNode->rootEditableElement()) {
            // Send BeforeTextInsertedEvent. The event handler will update text if necessary.
            Ref event = BeforeTextInsertedEvent::create(text);
            RefPtr { startNode->rootEditableElement() }->dispatchEvent(event);
            newText = event->text();
        }
    }
    return newText;
}

bool canAppendNewLineFeedToSelection(const VisibleSelection& selection)
{
    RefPtr node = selection.rootEditableElement();
    if (!node)
        return false;
    
    Ref event = BeforeTextInsertedEvent::create("\n"_s);
    node->dispatchEvent(event);
    return event->text().length();
}

}
