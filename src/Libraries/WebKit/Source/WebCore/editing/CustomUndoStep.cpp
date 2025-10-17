/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 9, 2021.
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
#include "CustomUndoStep.h"

#include "Document.h"
#include "UndoItem.h"
#include "UndoManager.h"
#include "VoidCallback.h"

namespace WebCore {

CustomUndoStep::CustomUndoStep(UndoItem& item)
    : m_undoItem(item)
{
}

void CustomUndoStep::unapply()
{
    if (!isValid())
        return;

    // FIXME: It's currently unclear how input events should be dispatched when unapplying or reapplying custom
    // edit commands. Should the page be allowed to specify a target in the DOM for undo and redo?
    Ref<UndoItem> protectedUndoItem(*m_undoItem);
    protectedUndoItem->protectedDocument()->updateLayoutIgnorePendingStylesheets();
    protectedUndoItem->undoHandler().handleEvent();
}

void CustomUndoStep::reapply()
{
    if (!isValid())
        return;

    Ref<UndoItem> protectedUndoItem(*m_undoItem);
    protectedUndoItem->protectedDocument()->updateLayoutIgnorePendingStylesheets();
    protectedUndoItem->redoHandler().handleEvent();
}

bool CustomUndoStep::isValid() const
{
    return m_undoItem && m_undoItem->isValid();
}

String CustomUndoStep::label() const
{
    if (!isValid()) {
        ASSERT_NOT_REACHED();
        return emptyString();
    }
    return m_undoItem->label();
}

void CustomUndoStep::didRemoveFromUndoManager()
{
    if (RefPtr undoItem = std::exchange(m_undoItem, nullptr).get())
        undoItem->invalidate();
}

} // namespace WebCore
