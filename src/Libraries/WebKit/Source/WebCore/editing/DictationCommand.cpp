/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 11, 2024.
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
#include "DictationCommand.h"

#include "AlternativeTextController.h"
#include "Document.h"
#include "DocumentInlines.h"
#include "DocumentMarkerController.h"
#include "FrameDestructionObserverInlines.h"
#include "FrameSelection.h"
#include "InsertParagraphSeparatorCommand.h"
#include "InsertTextCommand.h"
#include "LocalFrame.h"
#include "Text.h"

namespace WebCore {

class DictationCommandLineOperation {
public:
    DictationCommandLineOperation(DictationCommand& dictationCommand)
        : m_dictationCommand(dictationCommand)
    { }
    
    void operator()(size_t lineOffset, size_t lineLength, bool isLastLine) const
    {
        if (lineLength > 0)
            Ref { m_dictationCommand.get() }->insertTextRunWithoutNewlines(lineOffset, lineLength);
        if (!isLastLine)
            Ref { m_dictationCommand.get() }->insertParagraphSeparator();
    }
private:
    WeakRef<DictationCommand> m_dictationCommand;
};

class DictationMarkerSupplier : public TextInsertionMarkerSupplier {
public:
    static Ref<DictationMarkerSupplier> create(Vector<DictationAlternative>&& alternatives)
    {
        return adoptRef(*new DictationMarkerSupplier(WTFMove(alternatives)));
    }

    void addMarkersToTextNode(Text& textNode, unsigned offsetOfInsertion, const String& textToBeInserted) override
    {
        Ref document = textNode.document();
        auto& markerController = document->markers();
        for (auto& alternative : m_alternatives) {
            DocumentMarker::DictationData data { alternative.context, textToBeInserted.substring(alternative.range.location, alternative.range.length) };
            markerController.addMarker(textNode, alternative.range.location + offsetOfInsertion, alternative.range.length, DocumentMarkerType::DictationAlternatives, WTFMove(data));
            markerController.addMarker(textNode, alternative.range.location + offsetOfInsertion, alternative.range.length, DocumentMarkerType::SpellCheckingExemption);
        }
    }

protected:
    DictationMarkerSupplier(Vector<DictationAlternative>&& alternatives)
        : m_alternatives(WTFMove(alternatives))
    {
    }
private:
    Vector<DictationAlternative> m_alternatives;
};

DictationCommand::DictationCommand(Ref<Document>&& document, const String& text, const Vector<DictationAlternative>& alternatives)
    : TextInsertionBaseCommand(WTFMove(document))
    , m_textToInsert(text)
    , m_alternatives(alternatives)
{
}

void DictationCommand::insertText(Ref<Document>&& document, const String& text, const Vector<DictationAlternative>& alternatives, const VisibleSelection& selectionForInsertion)
{
    RefPtr frame { document->frame() };
    ASSERT(frame);

    VisibleSelection currentSelection = frame->selection().selection();

    String newText = dispatchBeforeTextInsertedEvent(text, selectionForInsertion, false);

    RefPtr<DictationCommand> cmd;
    if (newText == text)
        cmd = DictationCommand::create(WTFMove(document), newText, alternatives);
    else {
        // If the text was modified before insertion, the location of dictation alternatives
        // will not be valid anymore. We will just drop the alternatives.
        cmd = DictationCommand::create(WTFMove(document), newText, Vector<DictationAlternative>());
    }
    applyTextInsertionCommand(frame.get(), *cmd, selectionForInsertion, currentSelection);
}

void DictationCommand::doApply()
{
    DictationCommandLineOperation operation(*this);
    forEachLineInString(m_textToInsert, operation);
    postTextStateChangeNotification(AXTextEditTypeDictation, m_textToInsert);
}

void DictationCommand::insertTextRunWithoutNewlines(size_t lineStart, size_t lineLength)
{
    Vector<DictationAlternative> alternativesInLine;
    collectDictationAlternativesInRange(lineStart, lineLength, alternativesInLine);
    auto command = InsertTextCommand::createWithMarkerSupplier(protectedDocument(), m_textToInsert.substring(lineStart, lineLength), DictationMarkerSupplier::create(WTFMove(alternativesInLine)), EditAction::Dictation);
    applyCommandToComposite(WTFMove(command), endingSelection());
}

void DictationCommand::insertParagraphSeparator()
{
    if (!canAppendNewLineFeedToSelection(endingSelection()))
        return;

    applyCommandToComposite(InsertParagraphSeparatorCommand::create(protectedDocument(), false, false, EditAction::Dictation));
}

void DictationCommand::collectDictationAlternativesInRange(size_t rangeStart, size_t rangeLength, Vector<DictationAlternative>& alternatives)
{
    for (auto& alternative : m_alternatives) {
        if (alternative.range.location >= rangeStart && (alternative.range.location + alternative.range.length) <= rangeStart + rangeLength)
            alternatives.append({ { alternative.range.location - rangeStart, alternative.range.length }, alternative.context });
    }

}

}
