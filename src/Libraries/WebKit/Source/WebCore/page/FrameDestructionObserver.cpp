/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 4, 2024.
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
#include "FrameDestructionObserver.h"

#include "LocalFrame.h"

namespace WebCore {

FrameDestructionObserver::FrameDestructionObserver(LocalFrame* frame)
    : m_frame(nullptr)
{
    observeFrame(frame);
}

FrameDestructionObserver::~FrameDestructionObserver()
{
    observeFrame(nullptr);
}

void FrameDestructionObserver::observeFrame(LocalFrame* frame)
{
    if (m_frame)
        m_frame->removeDestructionObserver(*this);

    m_frame = frame;

    if (m_frame)
        m_frame->addDestructionObserver(*this);
}

void FrameDestructionObserver::frameDestroyed()
{
    m_frame = nullptr;
}

void FrameDestructionObserver::willDetachPage()
{
    // Subclasses should override this function to handle this notification.
}

}
