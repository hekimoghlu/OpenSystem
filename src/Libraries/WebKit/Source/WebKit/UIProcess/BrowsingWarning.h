/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 7, 2021.
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

#include <wtf/RefCounted.h>
#include <wtf/RetainPtr.h>
#include <wtf/URL.h>
#include <wtf/text/WTFString.h>

OBJC_CLASS NSAttributedString;
OBJC_CLASS NSURL;
OBJC_CLASS SSBServiceLookupResult;

namespace WebKit {

class BrowsingWarning : public RefCounted<BrowsingWarning> {
public:
    struct HTTPSNavigationFailureData { };

    struct SafeBrowsingWarningData {
#if PLATFORM(COCOA)
        RetainPtr<SSBServiceLookupResult> result;
#endif
    };

    using Data = std::variant<SafeBrowsingWarningData, HTTPSNavigationFailureData>;

#if HAVE(SAFE_BROWSING)
    static Ref<BrowsingWarning> create(const URL& url, bool forMainFrameNavigation, Data&& data)
    {
        return adoptRef(*new BrowsingWarning(url, forMainFrameNavigation, WTFMove(data)));
    }
#endif
#if PLATFORM(COCOA)
    static Ref<BrowsingWarning> create(URL&& url, String&& title, String&& warning, RetainPtr<NSAttributedString>&& details, Data&& data)
    {
        return adoptRef(*new BrowsingWarning(WTFMove(url), WTFMove(title), WTFMove(warning), WTFMove(details), WTFMove(data)));
    }
#endif

    const URL& url() const { return m_url; }
    const String& title() const { return m_title; }
    const String& warning() const { return m_warning; }
    bool forMainFrameNavigation() const { return m_forMainFrameNavigation; }
#if PLATFORM(COCOA)
    RetainPtr<NSAttributedString> details() const { return m_details; }
#endif
    const Data& data() const { return m_data; }

    static NSURL *visitUnsafeWebsiteSentinel();
    static NSURL *confirmMalwareSentinel();

private:
#if HAVE(SAFE_BROWSING)
    BrowsingWarning(const URL&, bool, Data&&);
#endif
#if PLATFORM(COCOA)
    BrowsingWarning(URL&&, String&&, String&&, RetainPtr<NSAttributedString>&&, Data&&);
#endif

    URL m_url;
    String m_title;
    String m_warning;
    bool m_forMainFrameNavigation { false };
#if PLATFORM(COCOA)
    RetainPtr<NSAttributedString> m_details;
#endif
    const Data m_data;
};

} // namespace WebKit
