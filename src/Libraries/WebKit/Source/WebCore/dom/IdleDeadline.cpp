/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 22, 2025.
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
#include "IdleDeadline.h"

#include "Document.h"
#include "LocalDOMWindow.h"
#include "Performance.h"
#include "WindowEventLoop.h"
#include <wtf/RefPtr.h>

namespace WebCore {

DOMHighResTimeStamp IdleDeadline::timeRemaining(Document& document) const
{
    RefPtr window { document.domWindow() };
    if (!window || m_didTimeout == DidTimeout::Yes)
        return 0;
    Ref performance = window->performance();
    auto now = performance->now();
    auto deadline = performance->relativeTimeFromTimeOriginInReducedResolution(document.windowEventLoop().computeIdleDeadline() - performance->timeResolution());
    return deadline < now ? 0 : deadline - now;
}

} // namespace WebCore
