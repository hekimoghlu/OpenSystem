/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 16, 2022.
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
#include "InspectorHistory.h"

#include "Node.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(InspectorHistory);
WTF_MAKE_TZONE_ALLOCATED_IMPL(InspectorHistory::Action);

class UndoableStateMark : public InspectorHistory::Action {
private:
    ExceptionOr<void> perform() final { return { }; }
    ExceptionOr<void> undo() final { return { }; }
    ExceptionOr<void> redo() final { return { }; }
    bool isUndoableStateMark() final { return true; }
};

ExceptionOr<void> InspectorHistory::perform(std::unique_ptr<Action> action)
{
    auto performResult = action->perform();
    if (performResult.hasException())
        return performResult.releaseException();

    if (!action->mergeId().isEmpty() && m_afterLastActionIndex > 0 && action->mergeId() == m_history[m_afterLastActionIndex - 1]->mergeId())
        m_history[m_afterLastActionIndex - 1]->merge(WTFMove(action));
    else {
        m_history.resize(m_afterLastActionIndex);
        m_history.append(WTFMove(action));
        ++m_afterLastActionIndex;
    }
    return { };
}

void InspectorHistory::markUndoableState()
{
    perform(makeUnique<UndoableStateMark>());
}

ExceptionOr<void> InspectorHistory::undo()
{
    while (m_afterLastActionIndex > 0 && m_history[m_afterLastActionIndex - 1]->isUndoableStateMark())
        --m_afterLastActionIndex;

    while (m_afterLastActionIndex > 0) {
        Action* action = m_history[m_afterLastActionIndex - 1].get();
        auto undoResult = action->undo();
        if (undoResult.hasException()) {
            reset();
            return undoResult.releaseException();
        }
        --m_afterLastActionIndex;
        if (action->isUndoableStateMark())
            break;
    }

    return { };
}

ExceptionOr<void> InspectorHistory::redo()
{
    while (m_afterLastActionIndex < m_history.size() && m_history[m_afterLastActionIndex]->isUndoableStateMark())
        ++m_afterLastActionIndex;

    while (m_afterLastActionIndex < m_history.size()) {
        Action* action = m_history[m_afterLastActionIndex].get();
        auto redoResult = action->redo();
        if (redoResult.hasException()) {
            reset();
            return redoResult.releaseException();
        }
        ++m_afterLastActionIndex;
        if (action->isUndoableStateMark())
            break;
    }
    return { };
}

void InspectorHistory::reset()
{
    m_afterLastActionIndex = 0;
    m_history.clear();
}

} // namespace WebCore
