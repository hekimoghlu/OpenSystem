/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 26, 2023.
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

#include "UserMessage.h"
#include <wtf/CompletionHandler.h>
#include <wtf/TZoneMallocInlines.h>

typedef struct OpaqueJSContext* JSGlobalContextRef;

namespace WebKit {
class DownloadProxy;
class WebKitWebResourceLoadManager;
}

namespace WKWPE {
class View;
}

namespace API {

class ViewClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(ViewClient);
public:
    virtual ~ViewClient() = default;

    virtual bool isGLibBasedAPI() { return false; }

    virtual void frameDisplayed(WKWPE::View&) { }
    virtual void willStartLoad(WKWPE::View&) { }
    virtual void didChangePageID(WKWPE::View&) { }
    virtual void didReceiveUserMessage(WKWPE::View&, WebKit::UserMessage&&, CompletionHandler<void(WebKit::UserMessage&&)>&& completionHandler) { completionHandler(WebKit::UserMessage()); }
    virtual WebKit::WebKitWebResourceLoadManager* webResourceLoadManager() { return nullptr; }

#if ENABLE(FULLSCREEN_API)
    virtual bool enterFullScreen(WKWPE::View&) { return false; };
    virtual bool exitFullScreen(WKWPE::View&) { return false; };
#endif
};

} // namespace API
