/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 28, 2023.
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
#include "SpellingCorrectionCommand.h"

#include "AlternativeTextController.h"
#include "DataTransfer.h"
#include "Document.h"
#include "DocumentFragment.h"
#include "Editor.h"
#include "LocalFrame.h"
#include "ReplaceSelectionCommand.h"
#include "SetSelectionCommand.h"
#include "StaticRange.h"
#include "TextIterator.h"
#include "markup.h"

namespace WebCore {

#if USE(AUTOCORRECTION_PANEL)
// On Mac OS X, we use this command to keep track of user undoing a correction for the first time.
// This information is needed by spell checking service to update user specific data.
class SpellingCorrectionRecordUndoCommand : public SimpleEditCommand {
public:
    static Ref<SpellingCorrectionRecordUndoCommand> create(Ref<Document>&& document, const String& corrected, const String& correction)
    {
        return adoptRef(*new SpellingCorrectionRecordUndoCommand(WTFMove(document), corrected, correction));
    }
private:
    SpellingCorrectionRecordUndoCommand(Ref<Document>&& document, const String& corrected, const String& correction)
        : SimpleEditCommand(WTFMove(document))
        , m_corrected(corrected)
        , m_correction(correction)
        , m_hasBeenUndone(false)
    {
    }

    void doApply() override
    {
    }

    void doUnapply() override
    {
        if (!m_hasBeenUndone) {
            protectedDocument()->protectedEditor()->unappliedSpellCorrection(startingSelection(), m_corrected, m_correction);
            m_hasBeenUndone = true;
        }
        
    }

#ifndef NDEBUG
    void getNodesInCommand(NodeSet&) override
    {
    }
#endif

    String m_corrected;
    String m_correction;
    bool m_hasBeenUndone;
};
#endif

SpellingCorrectionCommand::SpellingCorrectionCommand(const SimpleRange& rangeToBeCorrected, const String& correction)
    : CompositeEditCommand(rangeToBeCorrected.start.document(), EditAction::InsertReplacement)
    , m_rangeToBeCorrected(rangeToBeCorrected)
    , m_selectionToBeCorrected(m_rangeToBeCorrected)
    , m_correction(correction)
{
}

Ref<SpellingCorrectionCommand> SpellingCorrectionCommand::create(const SimpleRange& rangeToBeCorrected, const String& correction)
{
    return adoptRef(*new SpellingCorrectionCommand(rangeToBeCorrected, correction));
}

bool SpellingCorrectionCommand::willApplyCommand()
{
    m_correctionFragment = createFragmentFromText(m_rangeToBeCorrected, m_correction);
    return CompositeEditCommand::willApplyCommand();
}

void SpellingCorrectionCommand::doApply()
{
    m_corrected = plainText(m_rangeToBeCorrected);
    if (!m_corrected.length())
        return;

    Ref document = protectedDocument();
    if (!document->selection().shouldChangeSelection(m_selectionToBeCorrected))
        return;

    applyCommandToComposite(SetSelectionCommand::create(m_selectionToBeCorrected, FrameSelection::defaultSetSelectionOptions() | FrameSelection::SetSelectionOption::SpellCorrectionTriggered));
#if USE(AUTOCORRECTION_PANEL)
    applyCommandToComposite(SpellingCorrectionRecordUndoCommand::create(document.copyRef(), m_corrected, m_correction));
#endif

    applyCommandToComposite(ReplaceSelectionCommand::create(WTFMove(document), protectedCorrectionFragment(), ReplaceSelectionCommand::MatchStyle, EditAction::Paste));
}

String SpellingCorrectionCommand::inputEventData() const
{
    if (isEditingTextAreaOrTextInput())
        return m_correction;

    return CompositeEditCommand::inputEventData();
}

Vector<RefPtr<StaticRange>> SpellingCorrectionCommand::targetRanges() const
{
    return { 1, StaticRange::create(m_rangeToBeCorrected) };
}

RefPtr<DataTransfer> SpellingCorrectionCommand::inputEventDataTransfer() const
{
    if (!isEditingTextAreaOrTextInput())
        return DataTransfer::createForInputEvent(m_correction, serializeFragment(*protectedCorrectionFragment(), SerializedNodes::SubtreeIncludingNode));

    return CompositeEditCommand::inputEventDataTransfer();
}

bool SpellingCorrectionCommand::shouldRetainAutocorrectionIndicator() const
{
    return true;
}

} // namespace WebCore
