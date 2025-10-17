/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 23, 2022.
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

#if ENABLE(FULLSCREEN_API)

#include <wtf/Forward.h>

namespace WebCore {

class DeferredPromise;
class Document;
class Element;

class DocumentFullscreen {
public:
    static void exitFullscreen(Document&, RefPtr<DeferredPromise>&&);
    static bool fullscreenEnabled(Document&);

    WEBCORE_EXPORT static bool webkitFullscreenEnabled(Document&);
    WEBCORE_EXPORT static Element* webkitFullscreenElement(Document&);
    WEBCORE_EXPORT static void webkitExitFullscreen(Document&);
    WEBCORE_EXPORT static bool webkitIsFullScreen(Document&);
    WEBCORE_EXPORT static bool webkitFullScreenKeyboardInputAllowed(Document&);
    WEBCORE_EXPORT static Element* webkitCurrentFullScreenElement(Document&);
    WEBCORE_EXPORT static void webkitCancelFullScreen(Document&);
};

} // namespace WebCore

#endif // ENABLE(FULLSCREEN_API)
