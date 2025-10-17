/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 27, 2024.
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
#include "ReplaceRangeWithTextCommand.h"

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

ReplaceRangeWithTextCommand::ReplaceRangeWithTextCommand(const SimpleRange& rangeToBeReplaced, const String& text)
    : CompositeEditCommand(rangeToBeReplaced.start.document(), EditAction::InsertReplacement)
    , m_rangeToBeReplaced(rangeToBeReplaced)
    , m_text(text)
{
}

bool ReplaceRangeWithTextCommand::willApplyCommand()
{
    m_textFragment = createFragmentFromText(m_rangeToBeReplaced, m_text);
    return CompositeEditCommand::willApplyCommand();
}

void ReplaceRangeWithTextCommand::doApply()
{
    VisibleSelection selection { m_rangeToBeReplaced };

    if (!document().selection().shouldChangeSelection(selection))
        return;

    if (!characterCount(m_rangeToBeReplaced))
        return;

    applyCommandToComposite(SetSelectionCommand::create(selection, FrameSelection::defaultSetSelectionOptions()));
    applyCommandToComposite(ReplaceSelectionCommand::create(document(), m_textFragment.copyRef(), ReplaceSelectionCommand::MatchStyle, EditAction::Paste));
}

String ReplaceRangeWithTextCommand::inputEventData() const
{
    if (isEditingTextAreaOrTextInput())
        return m_text;

    return CompositeEditCommand::inputEventData();
}

RefPtr<DataTransfer> ReplaceRangeWithTextCommand::inputEventDataTransfer() const
{
    if (!isEditingTextAreaOrTextInput())
        return DataTransfer::createForInputEvent(m_text, serializeFragment(*protectedTextFragment(), SerializedNodes::SubtreeIncludingNode));

    return CompositeEditCommand::inputEventDataTransfer();
}

Vector<RefPtr<StaticRange>> ReplaceRangeWithTextCommand::targetRanges() const
{
    return { 1, StaticRange::create(m_rangeToBeReplaced) };
}

} // namespace WebCore
