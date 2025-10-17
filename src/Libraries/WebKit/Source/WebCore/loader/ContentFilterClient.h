/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 31, 2025.
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

#if ENABLE(CONTENT_FILTERING)

#include <wtf/AbstractRefCountedAndCanMakeWeakPtr.h>
#include <wtf/Forward.h>

namespace WebCore {
class ContentFilterClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::ContentFilterClient> : std::true_type { };
}

namespace WebCore {

class ContentFilterUnblockHandler;
class ResourceError;
class SharedBuffer;
class SubstituteData;

class ContentFilterClient : public AbstractRefCountedAndCanMakeWeakPtr<ContentFilterClient> {
public:
    virtual ~ContentFilterClient() = default;

    virtual void dataReceivedThroughContentFilter(const SharedBuffer&, size_t) = 0;
    virtual ResourceError contentFilterDidBlock(ContentFilterUnblockHandler, String&& unblockRequestDeniedScript) = 0;
    virtual void cancelMainResourceLoadForContentFilter(const ResourceError&) = 0;
    virtual void handleProvisionalLoadFailureFromContentFilter(const URL& blockedPageURL, SubstituteData&) = 0;
};

} // namespace WebCore

#endif // ENABLE(CONTENT_FILTERING)
