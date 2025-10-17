/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 16, 2022.
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

#include <wtf/CompletionHandler.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
class Cursor;
class IntRect;
class Region;
}

namespace WebKit {
class PlayStationWebView;
}

namespace API {

class ViewClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(ViewClient);
public:
    virtual ~ViewClient() = default;

    virtual void setViewNeedsDisplay(WebKit::PlayStationWebView&, const WebCore::Region&) { }
    virtual void enterFullScreen(WebKit::PlayStationWebView&, CompletionHandler<void(bool)>&& completionHandler) { completionHandler(false); }
    virtual void exitFullScreen(WebKit::PlayStationWebView&) { }
    virtual void closeFullScreen(WebKit::PlayStationWebView&) { }
    virtual void beganEnterFullScreen(WebKit::PlayStationWebView&, const WebCore::IntRect&, const WebCore::IntRect&) { }
    virtual void beganExitFullScreen(WebKit::PlayStationWebView&, const WebCore::IntRect&, const WebCore::IntRect&) { }
    virtual void setCursor(WebKit::PlayStationWebView& view, const WebCore::Cursor& cursor) { }
};

} // namespace API
