/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 7, 2022.
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
#import "WebExtensionContext.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#import "WKFrameInfoPrivate.h"
#import "WKWebViewPrivate.h"
#import "WebExtensionFrameIdentifier.h"
#import "WebExtensionFrameParameters.h"
#import "WebExtensionTab.h"
#import "WebExtensionUtilities.h"
#import "WebFrame.h"
#import "_WKFrameTreeNode.h"
#import <wtf/BlockPtr.h>

namespace WebKit {

static WebExtensionFrameParameters frameParametersForFrame(_WKFrameTreeNode *frame, _WKFrameTreeNode *parentFrame, WebExtensionTab* tab, WebExtensionContext* extensionContext, bool includeFrameIdentifier)
{
    auto *frameInfo = frame.info;
    auto frameURL = URL { frameInfo.request.URL };

    return {
        .errorOccurred = static_cast<bool>(frameInfo._errorOccurred),
        .url = extensionContext->hasPermission(frameURL, tab) ? std::optional { frameURL } : std::nullopt,
        .parentFrameIdentifier = parentFrame ? toWebExtensionFrameIdentifier(parentFrame.info) : WebExtensionFrameConstants::NoneIdentifier,
        .frameIdentifier = includeFrameIdentifier ? std::optional { toWebExtensionFrameIdentifier(frameInfo) } : std::nullopt,
        .documentIdentifier = WTF::UUID::fromNSUUID(frameInfo._documentIdentifier)
    };
}

bool WebExtensionContext::isWebNavigationMessageAllowed()
{
    return isLoaded() && hasPermission(WKWebExtensionPermissionWebNavigation);
}

void WebExtensionContext::webNavigationTraverseFrameTreeForFrame(_WKFrameTreeNode *frame, _WKFrameTreeNode *parentFrame, WebExtensionTab* tab, Vector<WebExtensionFrameParameters> &frames)
{
    frames.append(frameParametersForFrame(frame, parentFrame, tab, this, true));

    for (_WKFrameTreeNode *childFrame in frame.childFrames)
        webNavigationTraverseFrameTreeForFrame(childFrame, frame, tab, frames);
}

std::optional<WebExtensionFrameParameters> WebExtensionContext::webNavigationFindFrameIdentifierInFrameTree(_WKFrameTreeNode *frame, _WKFrameTreeNode *parentFrame, WebExtensionTab* tab, WebExtensionFrameIdentifier targetFrameIdentifier)
{
    if (toWebExtensionFrameIdentifier(frame.info) == targetFrameIdentifier)
        return frameParametersForFrame(frame, parentFrame, tab, this, false);

    for (_WKFrameTreeNode *childFrame in frame.childFrames) {
        if (auto matchingChildFrame = webNavigationFindFrameIdentifierInFrameTree(childFrame, frame, tab, targetFrameIdentifier))
            return matchingChildFrame;
    }

    return std::nullopt;
}

void WebExtensionContext::webNavigationGetFrame(WebExtensionTabIdentifier tabIdentifier, WebExtensionFrameIdentifier frameIdentifier, CompletionHandler<void(Expected<std::optional<WebExtensionFrameParameters>, WebExtensionError>&&)>&& completionHandler)
{
    RefPtr tab = getTab(tabIdentifier);
    if (!tab) {
        completionHandler(toWebExtensionError(@"webNavigation.getFrame()", nullString(), @"tab not found"));
        return;
    }

    auto *webView = tab->webView();
    if (!webView) {
        completionHandler(toWebExtensionError(@"webNavigation.getFrame()", nullString(), @"tab not found"));
        return;
    }

    [webView _frames:makeBlockPtr([this, protectedThis = Ref { *this }, completionHandler = WTFMove(completionHandler), tab, frameIdentifier](_WKFrameTreeNode *mainFrame) mutable {
        if (!mainFrame.info.isMainFrame) {
            RELEASE_LOG_INFO(Extensions, "Skipping frame traversal because the mainFrame is nil");
            completionHandler(toWebExtensionError(@"webNavigation.getFrame()", nullString(), @"main frame not found"));
            return;
        }

        if (auto frameParameters = webNavigationFindFrameIdentifierInFrameTree(mainFrame, nil, tab.get(), frameIdentifier))
            completionHandler(WTFMove(frameParameters));
        else
            completionHandler(toWebExtensionError(@"webNavigation.getFrame()", nullString(), @"frame not found"));
    }).get()];
}

void WebExtensionContext::webNavigationGetAllFrames(WebExtensionTabIdentifier tabIdentifier, CompletionHandler<void(Expected<Vector<WebExtensionFrameParameters>, WebExtensionError>&&)>&& completionHandler)
{
    RefPtr tab = getTab(tabIdentifier);
    if (!tab) {
        completionHandler(toWebExtensionError(@"webNavigation.getAllFrames()", nullString(), @"tab not found"));
        return;
    }

    auto *webView = tab->webView();
    if (!webView) {
        completionHandler(toWebExtensionError(@"webNavigation.getAllFrames()", nullString(), @"tab not found"));
        return;
    }

    [webView _frames:makeBlockPtr([this, protectedThis = Ref { *this }, completionHandler = WTFMove(completionHandler), tab](_WKFrameTreeNode *mainFrame) mutable {
        if (!mainFrame.info.isMainFrame) {
            RELEASE_LOG_INFO(Extensions, "Skipping frame traversal because the mainFrame is nil");
            completionHandler(toWebExtensionError(@"webNavigation.getAllFrames()", nullString(), @"main frame not found"));
            return;
        }

        Vector<WebExtensionFrameParameters> frameParameters;
        webNavigationTraverseFrameTreeForFrame(mainFrame, nil, tab.get(), frameParameters);

        completionHandler(WTFMove(frameParameters));
    }).get()];
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
