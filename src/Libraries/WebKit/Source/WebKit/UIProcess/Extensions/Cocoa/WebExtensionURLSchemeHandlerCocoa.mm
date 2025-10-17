/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 9, 2023.
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
#if !__has_feature(objc_arc)
#error This file requires ARC. Add the "-fobjc-arc" compiler flag for this file.
#endif

#import "config.h"
#import "WebExtensionURLSchemeHandler.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#import "APIData.h"
#import "APIError.h"
#import "APIFrameInfo.h"
#import "WKNSData.h"
#import "WKNSError.h"
#import "WKURLSchemeTaskInternal.h"
#import "WKWebViewConfigurationPrivate.h"
#import "WebExtension.h"
#import "WebExtensionContext.h"
#import "WebExtensionContextProxyMessages.h"
#import "WebExtensionController.h"
#import "WebPageProxy.h"
#import "WebProcessProxy.h"
#import "WebURLSchemeTask.h"
#import <wtf/BlockPtr.h>

namespace WebKit {

class WebPageProxy;

constexpr NSInteger noPermissionErrorCode = NSURLErrorNoPermissionsToReadFile;

WebExtensionURLSchemeHandler::WebExtensionURLSchemeHandler(WebExtensionController& controller)
    : m_webExtensionController(controller)
{
}

void WebExtensionURLSchemeHandler::platformStartTask(WebPageProxy& page, WebURLSchemeTask& task)
{
    auto *operation = [NSBlockOperation blockOperationWithBlock:makeBlockPtr([this, protectedThis = Ref { *this }, &task, protectedTask = Ref { task }, &page, protectedPage = Ref { page }]() {
        // If a frame is loading, the frame request URL will be an empty string, since the request is actually the frame URL being loaded.
        // In this case, consider the firstPartyForCookies() to be the document including the frame. This fails for nested frames, since
        // it is always the main frame URL, not the immediate parent frame.
        // FIXME: <rdar://problem/59193765> Remove this workaround when there is a way to know the proper parent frame.
        URL frameDocumentURL = task.frameInfo().request().url().isEmpty() ? task.request().firstPartyForCookies() : task.frameInfo().request().url();
        URL requestURL = task.request().url();

        if (!m_webExtensionController) {
            task.didComplete([NSError errorWithDomain:NSURLErrorDomain code:noPermissionErrorCode userInfo:nil]);
            return;
        }

        RefPtr extensionContext = m_webExtensionController->extensionContext(requestURL);
        if (!extensionContext) {
            // We need to return the same error here, as we do below for URLs that don't match web_accessible_resources.
            // Otherwise, a page tracking extension injected content and watching extension UUIDs across page loads can fingerprint
            // the user and know the same set of extensions are installed and enabled for this user and that website.
            task.didComplete([NSError errorWithDomain:NSURLErrorDomain code:noPermissionErrorCode userInfo:nil]);
            return;
        }

#if ENABLE(INSPECTOR_EXTENSIONS)
        // Chrome does not require devtools extensions to explicitly list resources as web_accessible_resources.
        if (!frameDocumentURL.protocolIs("inspector-resource"_s) && !protocolHostAndPortAreEqual(frameDocumentURL, requestURL))
#else
        if (!protocolHostAndPortAreEqual(frameDocumentURL, requestURL))
#endif
        {
            if (!extensionContext->extension().isWebAccessibleResource(requestURL, frameDocumentURL)) {
                task.didComplete([NSError errorWithDomain:NSURLErrorDomain code:noPermissionErrorCode userInfo:nil]);
                return;
            }
        }

        bool loadingExtensionMainFrame = false;
        if (task.frameInfo().isMainFrame() && requestURL == frameDocumentURL) {
            if (!extensionContext->isURLForThisExtension(page.configuration().requiredWebExtensionBaseURL())) {
                task.didComplete([NSError errorWithDomain:NSURLErrorDomain code:NSURLErrorResourceUnavailable userInfo:nil]);
                return;
            }

            loadingExtensionMainFrame = true;
        }

        RefPtr<API::Error> error;
        RefPtr resourceData = extensionContext->extension().resourceDataForPath(requestURL.path().toString(), error);
        if (!resourceData || error) {
            extensionContext->recordErrorIfNeeded(wrapper(error));
            task.didComplete([NSError errorWithDomain:NSURLErrorDomain code:NSURLErrorFileDoesNotExist userInfo:nil]);
            return;
        }

        if (loadingExtensionMainFrame) {
            if (auto tab = extensionContext->getTab(page.identifier()))
                extensionContext->addExtensionTabPage(page, *tab);
        }

        auto mimeType = extensionContext->extension().resourceMIMETypeForPath(requestURL.path().toString());
        resourceData = extensionContext->localizedResourceData(resourceData, mimeType);

        auto *urlResponse = [[NSHTTPURLResponse alloc] initWithURL:requestURL statusCode:200 HTTPVersion:nil headerFields:@{
            @"Access-Control-Allow-Origin": @"*",
            @"Content-Security-Policy": extensionContext->extension().contentSecurityPolicy(),
            @"Content-Length": @(resourceData->size()).stringValue,
            @"Content-Type": mimeType
        }];

        task.didReceiveResponse(urlResponse);
        task.didReceiveData(WebCore::SharedBuffer::create(resourceData->span()));
        task.didComplete({ });
    }).get()];

    m_operations.set(task, operation);

    [NSOperationQueue.mainQueue addOperation:operation];
}

void WebExtensionURLSchemeHandler::platformStopTask(WebPageProxy& page, WebURLSchemeTask& task)
{
    auto operation = m_operations.take(task);
    [operation cancel];
}

void WebExtensionURLSchemeHandler::platformTaskCompleted(WebURLSchemeTask& task)
{
    m_operations.remove(task);
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
