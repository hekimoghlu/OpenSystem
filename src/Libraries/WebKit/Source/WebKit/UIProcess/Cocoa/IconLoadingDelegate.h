/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 12, 2023.
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

#import "WKFoundation.h"

#import "APIIconLoadingClient.h"
#import <wtf/CheckedPtr.h>
#import <wtf/RetainPtr.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/WeakObjCPtr.h>

@class WKWebView;
@protocol _WKIconLoadingDelegate;

namespace WebKit {

class IconLoadingDelegate : public CanMakeCheckedPtr<IconLoadingDelegate> {
    WTF_MAKE_TZONE_ALLOCATED(IconLoadingDelegate);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(IconLoadingDelegate);
public:
    explicit IconLoadingDelegate(WKWebView *);
    ~IconLoadingDelegate();

    std::unique_ptr<API::IconLoadingClient> createIconLoadingClient();

    RetainPtr<id <_WKIconLoadingDelegate> > delegate();
    void setDelegate(id <_WKIconLoadingDelegate>);

private:
    class IconLoadingClient : public API::IconLoadingClient {
        WTF_MAKE_TZONE_ALLOCATED(IconLoadingClient);
    public:
        explicit IconLoadingClient(IconLoadingDelegate&);
        ~IconLoadingClient();

    private:
        void getLoadDecisionForIcon(const WebCore::LinkIcon&, CompletionHandler<void(CompletionHandler<void(API::Data*)>&&)>&&) override;

        CheckedRef<IconLoadingDelegate> m_iconLoadingDelegate;
    };

    WKWebView *m_webView;
    WeakObjCPtr<id <_WKIconLoadingDelegate> > m_delegate;

    struct {
        bool webViewShouldLoadIconWithParametersCompletionHandler : 1;
    } m_delegateMethods;
};

} // namespace WebKit
