/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 23, 2025.
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
#include "FrameLoaderStateMachine.h"

#include <wtf/Assertions.h>

namespace WebCore {

    
FrameLoaderStateMachine::FrameLoaderStateMachine() 
    : m_state(CreatingInitialEmptyDocument)
{ 
}
    
bool FrameLoaderStateMachine::committingFirstRealLoad() const 
{
    return m_state == DisplayingInitialEmptyDocument;
}

bool FrameLoaderStateMachine::committedFirstRealDocumentLoad() const 
{
    return m_state >= DisplayingInitialEmptyDocumentPostCommit;
}

bool FrameLoaderStateMachine::creatingInitialEmptyDocument() const 
{
    return m_state == CreatingInitialEmptyDocument;
}

bool FrameLoaderStateMachine::isDisplayingInitialEmptyDocument() const 
{
    return m_state == DisplayingInitialEmptyDocument || m_state == DisplayingInitialEmptyDocumentPostCommit;
}

bool FrameLoaderStateMachine::firstLayoutDone() const
{
    return m_state == FirstLayoutDone;
}

void FrameLoaderStateMachine::advanceTo(State state)
{
    ASSERT(State(m_state + 1) == state || (firstLayoutDone() && state == CommittedFirstRealLoad));
    m_state = state;
}

} // namespace WebCore
