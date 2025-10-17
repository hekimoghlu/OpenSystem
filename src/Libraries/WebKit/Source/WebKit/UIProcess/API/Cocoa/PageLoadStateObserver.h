/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 25, 2022.
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
#import "PageLoadState.h"
#import <wtf/TZoneMallocInlines.h>
#import <wtf/WeakObjCPtr.h>

namespace WebKit {

class PageLoadStateObserver : public PageLoadState::Observer {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(PageLoadStateObserver);
public:
    PageLoadStateObserver(id object, NSString *activeURLKey = @"activeURL")
        : m_object(object)
        , m_activeURLKey(adoptNS([activeURLKey copy]))
    {
    }

    void ref() const final;
    void deref() const final;

    void clearObject()
    {
        m_object = nil;
    }

private:
    void willChangeIsLoading() override
    {
        [m_object.get() willChangeValueForKey:@"loading"];
    }

    void didChangeIsLoading() override
    {
        [m_object.get() didChangeValueForKey:@"loading"];
    }

    void willChangeTitle() override
    {
        [m_object.get() willChangeValueForKey:@"title"];
    }

    void didChangeTitle() override
    {
        [m_object.get() didChangeValueForKey:@"title"];
    }

    void willChangeActiveURL() override
    {
        [m_object.get() willChangeValueForKey:m_activeURLKey.get()];
    }

    void didChangeActiveURL() override
    {
        [m_object.get() didChangeValueForKey:m_activeURLKey.get()];
    }

    void willChangeHasOnlySecureContent() override
    {
        [m_object.get() willChangeValueForKey:@"hasOnlySecureContent"];
    }

    void didChangeHasOnlySecureContent() override
    {
        [m_object.get() didChangeValueForKey:@"hasOnlySecureContent"];
    }

    void willChangeEstimatedProgress() override
    {
        [m_object.get() willChangeValueForKey:@"estimatedProgress"];
    }

    void didChangeEstimatedProgress() override
    {
        [m_object.get() didChangeValueForKey:@"estimatedProgress"];
    }

    void willChangeCanGoBack() override { }
    void didChangeCanGoBack() override { }
    void willChangeCanGoForward() override { }
    void didChangeCanGoForward() override { }
    void willChangeNetworkRequestsInProgress() override { }
    void didChangeNetworkRequestsInProgress() override { }
    void willChangeCertificateInfo() override { }
    void didChangeCertificateInfo() override { }
    void didSwapWebProcesses() override { }

    void willChangeWebProcessIsResponsive() override
    {
        [m_object.get() willChangeValueForKey:@"_webProcessIsResponsive"];
    }

    void didChangeWebProcessIsResponsive() override
    {
        [m_object.get() didChangeValueForKey:@"_webProcessIsResponsive"];
    }

    WeakObjCPtr<id> m_object;
    RetainPtr<NSString> m_activeURLKey;
};

}
