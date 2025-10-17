/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 23, 2025.
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
#include "WebKitFormClient.h"

#include "APIFormClient.h"
#include "WebFormSubmissionListenerProxy.h"
#include "WebKitFormSubmissionRequestPrivate.h"
#include "WebKitWebViewPrivate.h"
#include <wtf/glib/GRefPtr.h>
#include <wtf/glib/WTFGType.h>

using namespace WebKit;

class FormClient final : public API::FormClient {
public:
    explicit FormClient(WebKitWebView* webView)
        : m_webView(webView)
    {
    }

private:
    void willSubmitForm(WebPageProxy&, WebFrameProxy&, WebFrameProxy&, const Vector<std::pair<String, String>>& values, API::Object*, CompletionHandler<void()>&& completionHandler) override
    {
        GRefPtr<WebKitFormSubmissionRequest> request = adoptGRef(webkitFormSubmissionRequestCreate(values, WebFormSubmissionListenerProxy::create(WTFMove(completionHandler))));
        webkitWebViewSubmitFormRequest(m_webView, request.get());
    }

    WebKitWebView* m_webView;
};

void attachFormClientToView(WebKitWebView* webView)
{
    webkitWebViewGetPage(webView).setFormClient(makeUnique<FormClient>(webView));
}
