/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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
#include "SetSelectionCommand.h"

#include "CompositeEditCommand.h"
#include "Document.h"
#include "LocalFrame.h"

namespace WebCore {

SetSelectionCommand::SetSelectionCommand(const VisibleSelection& selection, OptionSet<FrameSelection::SetSelectionOption> options)
    : SimpleEditCommand(selection.base().anchorNode()->document())
    , m_options(options)
    , m_selectionToSet(selection)
{
}

void SetSelectionCommand::doApply()
{
    FrameSelection& selection = document().selection();

    if (selection.shouldChangeSelection(m_selectionToSet) && !m_selectionToSet.isNoneOrOrphaned()) {
        selection.setSelection(m_selectionToSet, m_options);
        setEndingSelection(m_selectionToSet);
    }
}

void SetSelectionCommand::doUnapply()
{
    FrameSelection& selection = document().selection();

    if (selection.shouldChangeSelection(startingSelection()) && !startingSelection().isNoneOrOrphaned())
        selection.setSelection(startingSelection(), m_options);
}

} // namespace WebCore
