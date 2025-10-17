/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 7, 2022.
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

#include <wtf/Noncopyable.h>

namespace WebCore {

// Encapsulates a state machine for FrameLoader. Note that this is different from FrameState,
// which stores the state of the current load that FrameLoader is executing.
class FrameLoaderStateMachine {
    WTF_MAKE_NONCOPYABLE(FrameLoaderStateMachine);
public:
    FrameLoaderStateMachine();

    // Once a load has been committed, the state may
    // alternate between CommittedFirstRealLoad and FirstLayoutDone.
    // Otherwise, the states only go down the list.
    enum State {
        CreatingInitialEmptyDocument,
        DisplayingInitialEmptyDocument,
        DisplayingInitialEmptyDocumentPostCommit,
        CommittedFirstRealLoad,
        FirstLayoutDone
    };

    WEBCORE_EXPORT bool committingFirstRealLoad() const;
    bool committedFirstRealDocumentLoad() const;
    WEBCORE_EXPORT bool creatingInitialEmptyDocument() const;
    WEBCORE_EXPORT bool isDisplayingInitialEmptyDocument() const;
    WEBCORE_EXPORT bool firstLayoutDone() const;
    void advanceTo(State);

    State stateForDebugging() const { return m_state; }

private:
    State m_state;
};

} // namespace WebCore
