/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 20, 2025.
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
#include "DefaultUndoController.h"

#include "UndoOrRedo.h"
#include "WebEditCommandProxy.h"
#include <wtf/RefPtr.h>

namespace WebKit {

void DefaultUndoController::registerEditCommand(Ref<WebEditCommandProxy>&& command, UndoOrRedo undoOrRedo)
{
    if (undoOrRedo == UndoOrRedo::Undo)
        m_undoStack.append(WTFMove(command));
    else
        m_redoStack.append(WTFMove(command));
}

void DefaultUndoController::clearAllEditCommands()
{
    m_undoStack.clear();
    m_redoStack.clear();
}

bool DefaultUndoController::canUndoRedo(UndoOrRedo undoOrRedo)
{
    if (undoOrRedo == UndoOrRedo::Undo)
        return !m_undoStack.isEmpty();

    return !m_redoStack.isEmpty();
}

void DefaultUndoController::executeUndoRedo(UndoOrRedo undoOrRedo)
{
    RefPtr<WebEditCommandProxy> command;
    if (undoOrRedo == UndoOrRedo::Undo) {
        command = m_undoStack.last();
        m_undoStack.removeLast();
        command->unapply();
    } else {
        command = m_redoStack.last();
        m_redoStack.removeLast();
        command->reapply();
    }
}

} // namespace WebKit
