/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 17, 2023.
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
#import "FindClient.h"

#import "_WKFindDelegate.h"
#import <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FindClient);

FindClient::FindClient(WKWebView *webView)
    : m_webView(webView)
{
}
    
RetainPtr<id <_WKFindDelegate>> FindClient::delegate()
{
    return m_delegate.get();
}

void FindClient::setDelegate(id <_WKFindDelegate> delegate)
{
    m_delegate = delegate;

    m_delegateMethods.webviewDidCountStringMatches = [delegate respondsToSelector:@selector(_webView:didCountMatches:forString:)];
    m_delegateMethods.webviewDidFindString = [delegate respondsToSelector:@selector(_webView:didFindMatches:forString:withMatchIndex:)];
    m_delegateMethods.webviewDidFailToFindString = [delegate respondsToSelector:@selector(_webView:didFailToFindString:)];
    m_delegateMethods.webviewDidAddLayerForFindOverlay = [delegate respondsToSelector:@selector(_webView:didAddLayerForFindOverlay:)];
    m_delegateMethods.webviewDidRemoveLayerForFindOverlay = [delegate respondsToSelector:@selector(_webViewDidRemoveLayerForFindOverlay:)];
}
    
void FindClient::didCountStringMatches(WebPageProxy*, const String& string, uint32_t matchCount)
{
    if (m_delegateMethods.webviewDidCountStringMatches)
        [m_delegate.get() _webView:m_webView didCountMatches:matchCount forString:string];
}

void FindClient::didFindString(WebPageProxy*, const String& string, const Vector<WebCore::IntRect>&, uint32_t matchCount, int32_t matchIndex, bool)
{
    if (m_delegateMethods.webviewDidFindString)
        [m_delegate.get() _webView:m_webView didFindMatches:matchCount forString:string withMatchIndex:matchIndex];
}

void FindClient::didFailToFindString(WebPageProxy*, const String& string)
{
    if (m_delegateMethods.webviewDidFailToFindString)
        [m_delegate.get() _webView:m_webView didFailToFindString:string];
}

void FindClient::didAddLayerForFindOverlay(WebKit::WebPageProxy*, PlatformLayer* layer)
{
    if (m_delegateMethods.webviewDidAddLayerForFindOverlay)
        [m_delegate _webView:m_webView didAddLayerForFindOverlay:layer];
}

void FindClient::didRemoveLayerForFindOverlay(WebKit::WebPageProxy*)
{
    if (m_delegateMethods.webviewDidRemoveLayerForFindOverlay)
        [m_delegate _webViewDidRemoveLayerForFindOverlay:m_webView];
}

} // namespace WebKit
