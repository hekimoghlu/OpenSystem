/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 9, 2021.
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

#include <wtf/text/WTFString.h>

namespace WebKit {

class WebInspectorUIProxy;

class WebInspectorUIProxyClient {
public:
    virtual ~WebInspectorUIProxyClient() = default;
    virtual bool openWindow(WebInspectorUIProxy&) = 0;
    virtual void didClose(WebInspectorUIProxy&) = 0;
    virtual bool bringToFront(WebInspectorUIProxy&) = 0;
    virtual void inspectedURLChanged(WebInspectorUIProxy&, const String& url) = 0;
    virtual bool attach(WebInspectorUIProxy&) = 0;
    virtual bool detach(WebInspectorUIProxy&) = 0;
    virtual void didChangeAttachedHeight(WebInspectorUIProxy&, unsigned height) = 0;
    virtual void didChangeAttachedWidth(WebInspectorUIProxy&, unsigned width) = 0;
    virtual void didChangeAttachAvailability(WebInspectorUIProxy&, bool available) = 0;
};

} // namespace WebKit
