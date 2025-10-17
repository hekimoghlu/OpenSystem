/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 4, 2023.
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

#if ENABLE(WK_WEB_EXTENSIONS)

#include "WebURLSchemeHandler.h"
#include <wtf/Forward.h>
#include <wtf/RetainPtr.h>
#include <wtf/WeakPtr.h>

OBJC_CLASS NSBlockOperation;

namespace WebKit {

class WebExtensionController;

class WebExtensionURLSchemeHandler : public WebURLSchemeHandler {
public:
    static Ref<WebExtensionURLSchemeHandler> create(WebExtensionController& controller)
    {
        return adoptRef(*new WebExtensionURLSchemeHandler(controller));
    }

private:
    WebExtensionURLSchemeHandler(WebExtensionController&);

    void platformStartTask(WebPageProxy&, WebURLSchemeTask&) final;
    void platformStopTask(WebPageProxy&, WebURLSchemeTask&) final;
    void platformTaskCompleted(WebURLSchemeTask&) final;

    WeakPtr<WebExtensionController> m_webExtensionController;
    HashMap<Ref<WebURLSchemeTask>, RetainPtr<NSBlockOperation>> m_operations;
}; // class WebExtensionURLSchemeHandler

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
