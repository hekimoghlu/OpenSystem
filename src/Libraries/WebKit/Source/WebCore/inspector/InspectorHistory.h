/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 19, 2023.
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
#pragma once

#include "ExceptionOr.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class InspectorHistory final {
    WTF_MAKE_TZONE_ALLOCATED(InspectorHistory);
    WTF_MAKE_NONCOPYABLE(InspectorHistory);
public:
    class Action {
        WTF_MAKE_TZONE_ALLOCATED(Action);
    public:
        virtual ~Action() = default;

        virtual String mergeId() { return emptyString(); }
        virtual void merge(std::unique_ptr<Action>) { };

        virtual ExceptionOr<void> perform() = 0;

        virtual ExceptionOr<void> undo() = 0;
        virtual ExceptionOr<void> redo() = 0;

        virtual bool isUndoableStateMark() { return false; }

    private:
        String m_name;
    };

    InspectorHistory() = default;

    ExceptionOr<void> perform(std::unique_ptr<Action>);
    void markUndoableState();

    ExceptionOr<void> undo();
    ExceptionOr<void> redo();
    void reset();

private:
    Vector<std::unique_ptr<Action>> m_history;
    size_t m_afterLastActionIndex { 0 };
};

} // namespace WebCore
