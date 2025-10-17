/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 29, 2024.
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
#include "WebEditorClient.h"

#include <WebCore/CompositionHighlight.h>
#include <WebCore/Document.h>
#include <WebCore/Editor.h>
#include <WebCore/FrameDestructionObserverInlines.h>
#include <WebCore/KeyboardEvent.h>
#include <WebCore/LocalFrame.h>
#include <WebCore/PlatformKeyboardEvent.h>
#include <WebPage.h>

namespace WebKit {
using namespace WebCore;

void WebEditorClient::handleInputMethodKeydown(KeyboardEvent& event)
{
    auto* platformEvent = event.underlyingPlatformEvent();
    if (platformEvent && platformEvent->handledByInputMethod())
        event.setDefaultHandled();
}

void WebEditorClient::didDispatchInputMethodKeydown(KeyboardEvent& event)
{
    auto* platformEvent = event.underlyingPlatformEvent();
    ASSERT(event.target());
    RefPtr frame = downcast<Node>(event.target())->document().frame();
    ASSERT(frame);

    if (const auto& underlines = platformEvent->preeditUnderlines()) {
        auto rangeStart = platformEvent->preeditSelectionRangeStart().value_or(0);
        auto rangeLength = platformEvent->preeditSelectionRangeLength().value_or(0);
        frame->editor().setComposition(platformEvent->text(), underlines.value(), { }, { }, rangeStart, rangeStart + rangeLength);
    } else
        frame->editor().confirmComposition(platformEvent->text());
}

} // namespace WebKit
