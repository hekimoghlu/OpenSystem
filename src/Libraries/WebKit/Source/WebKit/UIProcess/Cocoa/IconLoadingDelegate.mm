/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 30, 2025.
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
#import "IconLoadingDelegate.h"

#import "WKNSData.h"
#import "_WKIconLoadingDelegate.h"
#import "_WKLinkIconParametersInternal.h"
#import <wtf/BlockPtr.h>
#import <wtf/RunLoop.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(IconLoadingDelegate);

IconLoadingDelegate::IconLoadingDelegate(WKWebView *webView)
    : m_webView(webView)
{
}

IconLoadingDelegate::~IconLoadingDelegate()
{
}

std::unique_ptr<API::IconLoadingClient> IconLoadingDelegate::createIconLoadingClient()
{
    return makeUnique<IconLoadingClient>(*this);
}

RetainPtr<id <_WKIconLoadingDelegate> > IconLoadingDelegate::delegate()
{
    return m_delegate.get();
}

void IconLoadingDelegate::setDelegate(id <_WKIconLoadingDelegate> delegate)
{
    m_delegate = delegate;

    m_delegateMethods.webViewShouldLoadIconWithParametersCompletionHandler = [delegate respondsToSelector:@selector(webView:shouldLoadIconWithParameters:completionHandler:)];
}

IconLoadingDelegate::IconLoadingClient::IconLoadingClient(IconLoadingDelegate& iconLoadingDelegate)
    : m_iconLoadingDelegate(iconLoadingDelegate)
{
}

IconLoadingDelegate::IconLoadingClient::~IconLoadingClient()
{
}

typedef void (^IconLoadCompletionHandler)(NSData*);

WTF_MAKE_TZONE_ALLOCATED_IMPL(IconLoadingDelegate::IconLoadingClient);

void IconLoadingDelegate::IconLoadingClient::getLoadDecisionForIcon(const WebCore::LinkIcon& linkIcon, CompletionHandler<void(CompletionHandler<void(API::Data*)>&&)>&& completionHandler)
{
    if (!m_iconLoadingDelegate->m_delegateMethods.webViewShouldLoadIconWithParametersCompletionHandler) {
        completionHandler(nullptr);
        return;
    }

    auto delegate = m_iconLoadingDelegate->m_delegate.get();
    if (!delegate) {
        completionHandler(nullptr);
        return;
    }

    RetainPtr<_WKLinkIconParameters> parameters = adoptNS([[_WKLinkIconParameters alloc] _initWithLinkIcon:linkIcon]);

    [delegate webView:m_iconLoadingDelegate->m_webView shouldLoadIconWithParameters:parameters.get() completionHandler:makeBlockPtr([completionHandler = WTFMove(completionHandler)] (IconLoadCompletionHandler loadCompletionHandler) mutable {
        ASSERT(RunLoop::isMain());
        if (loadCompletionHandler) {
            completionHandler([loadCompletionHandler = makeBlockPtr(loadCompletionHandler)](API::Data* data) {
                loadCompletionHandler(wrapper(data));
            });
        } else
            completionHandler(nullptr);
    }).get()];
}

} // namespace WebKit
