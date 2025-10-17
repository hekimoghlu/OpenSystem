/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 21, 2023.
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
#import "config.h"
#import "WebURLSchemeHandlerCocoa.h"

#import "WKFoundation.h"
#import "WKURLSchemeHandler.h"
#import "WKURLSchemeTaskInternal.h"
#import "WKWebViewInternal.h"
#import "WebPageProxy.h"
#import "WebURLSchemeTask.h"
#import <wtf/RunLoop.h>

namespace WebKit {

Ref<WebURLSchemeHandlerCocoa> WebURLSchemeHandlerCocoa::create(id <WKURLSchemeHandler> apiHandler)
{
    return adoptRef(*new WebURLSchemeHandlerCocoa(apiHandler));
}

WebURLSchemeHandlerCocoa::WebURLSchemeHandlerCocoa(id <WKURLSchemeHandler> apiHandler)
    : m_apiHandler(apiHandler)
{
}

void WebURLSchemeHandlerCocoa::platformStartTask(WebPageProxy& page, WebURLSchemeTask& task)
{
    auto strongTask = retainPtr(wrapper(task));
    if (auto webView = page.cocoaView())
        [m_apiHandler.get() webView:webView.get() startURLSchemeTask:strongTask.get()];
}

void WebURLSchemeHandlerCocoa::platformStopTask(WebPageProxy& page, WebURLSchemeTask& task)
{
    auto strongTask = retainPtr(wrapper(task));
    if (auto webView = page.cocoaView())
        [m_apiHandler.get() webView:webView.get() stopURLSchemeTask:strongTask.get()];
    else
        task.suppressTaskStoppedExceptions();
}

} // namespace WebKit
