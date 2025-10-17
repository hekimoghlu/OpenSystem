/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 19, 2022.
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
#ifndef FindClient_h
#define FindClient_h

#import "WKFoundation.h"

#import "APIFindClient.h"
#import <wtf/TZoneMalloc.h>
#import <wtf/WeakObjCPtr.h>

@class WKWebView;
@protocol _WKFindDelegate;

namespace WebKit {

class FindClient final : public API::FindClient {
    WTF_MAKE_TZONE_ALLOCATED(FindClient);
public:
    explicit FindClient(WKWebView *);
    
    RetainPtr<id <_WKFindDelegate>> delegate();
    void setDelegate(id <_WKFindDelegate>);
    
private:
    // From API::FindClient
    virtual void didCountStringMatches(WebPageProxy*, const String&, uint32_t matchCount);
    virtual void didFindString(WebPageProxy*, const String&, const Vector<WebCore::IntRect>&, uint32_t matchCount, int32_t matchIndex, bool didWrapAround);
    virtual void didFailToFindString(WebPageProxy*, const String&);

    virtual void didAddLayerForFindOverlay(WebKit::WebPageProxy*, CALayer *);
    virtual void didRemoveLayerForFindOverlay(WebKit::WebPageProxy*);
    
    WKWebView *m_webView;
    WeakObjCPtr<id <_WKFindDelegate>> m_delegate;
    
    struct {
        bool webviewDidCountStringMatches : 1;
        bool webviewDidFindString : 1;
        bool webviewDidFailToFindString : 1;
        bool webviewDidAddLayerForFindOverlay : 1;
        bool webviewDidRemoveLayerForFindOverlay : 1;
    } m_delegateMethods;
};
    
} // namespace WebKit

#endif // FindClient_h
