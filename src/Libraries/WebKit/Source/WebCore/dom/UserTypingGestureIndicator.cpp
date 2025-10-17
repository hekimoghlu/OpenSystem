/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 19, 2025.
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
#include "UserTypingGestureIndicator.h"

#include "Element.h"
#include "LocalFrame.h"
#include <wtf/NeverDestroyed.h>

namespace WebCore {

static bool s_processingUserTypingGesture;
bool UserTypingGestureIndicator::processingUserTypingGesture()
{
    return s_processingUserTypingGesture;
}

static RefPtr<Node>& focusedNode()
{
    static NeverDestroyed<RefPtr<Node>> node;
    return node;
}

Node* UserTypingGestureIndicator::focusedElementAtGestureStart()
{
    return focusedNode().get();
}

UserTypingGestureIndicator::UserTypingGestureIndicator(LocalFrame& frame)
    : m_previousProcessingUserTypingGesture(s_processingUserTypingGesture)
    , m_previousFocusedNode(focusedNode())
{
    s_processingUserTypingGesture = true;
    RefPtr document = frame.document();
    focusedNode() = document ? document->focusedElement() : nullptr;
}

UserTypingGestureIndicator::~UserTypingGestureIndicator()
{
    s_processingUserTypingGesture = m_previousProcessingUserTypingGesture;
    focusedNode() = m_previousFocusedNode;
}

} // namespace WebCore
