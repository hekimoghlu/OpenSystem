/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 10, 2022.
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
#include "DocumentFullscreen.h"

#if ENABLE(FULLSCREEN_API)
#include "DocumentInlines.h"
#include "Document.h"
#include "Element.h"
#include "EventLoop.h"
#include "FullscreenManager.h"
#include "JSDOMPromiseDeferred.h"

namespace WebCore {

bool DocumentFullscreen::webkitFullscreenEnabled(Document& document)
{
    return document.fullscreenManager().isFullscreenEnabled();
}

Element* DocumentFullscreen::webkitFullscreenElement(Document& document)
{
    return document.ancestorElementInThisScope(document.fullscreenManager().protectedFullscreenElement().get());
}

bool DocumentFullscreen::webkitIsFullScreen(Document& document)
{
    return document.fullscreenManager().isFullscreen();
}

bool DocumentFullscreen::webkitFullScreenKeyboardInputAllowed(Document& document)
{
    return document.fullscreenManager().isFullscreenKeyboardInputAllowed();
}

Element* DocumentFullscreen::webkitCurrentFullScreenElement(Document& document)
{
    return document.ancestorElementInThisScope(document.fullscreenManager().protectedCurrentFullscreenElement().get());
}

void DocumentFullscreen::webkitCancelFullScreen(Document& document)
{
    document.fullscreenManager().cancelFullscreen();
}

// https://fullscreen.spec.whatwg.org/#exit-fullscreen
void DocumentFullscreen::exitFullscreen(Document& document, RefPtr<DeferredPromise>&& promise)
{
    if (!document.isFullyActive() || !document.fullscreenManager().fullscreenElement()) {
        promise->reject(Exception { ExceptionCode::TypeError, "Not in fullscreen"_s });
        return;
    }
    document.checkedFullscreenManager()->exitFullscreen(WTFMove(promise));
}

void DocumentFullscreen::webkitExitFullscreen(Document& document)
{
    if (document.fullscreenManager().fullscreenElement())
        document.checkedFullscreenManager()->exitFullscreen(nullptr);
}

// https://fullscreen.spec.whatwg.org/#dom-document-fullscreenenabled
bool DocumentFullscreen::fullscreenEnabled(Document& document)
{
    if (!document.isFullyActive())
        return false;
    return document.checkedFullscreenManager()->isFullscreenEnabled();
}

} // namespace WebCore

#endif // ENABLE(FULLSCREEN_API)
