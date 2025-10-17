/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 8, 2025.
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

#include "ResourceResponseBase.h"
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>

OBJC_CLASS NSURLResponse;

namespace WebCore {

class ResourceResponse : public ResourceResponseBase {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ResourceResponse, WEBCORE_EXPORT);
public:
    ResourceResponse()
    {
        m_initLevel = AllFields;
    }
    
    ResourceResponse(ResourceResponseBase&& base)
        : ResourceResponseBase(WTFMove(base))
    {
        m_initLevel = AllFields;
    }

    ResourceResponse(NSURLResponse *nsResponse)
        : m_nsResponse(nsResponse)
    {
        m_initLevel = Uninitialized;
        m_isNull = !nsResponse;
    }

    ResourceResponse(const URL& url, const String& mimeType, long long expectedLength, const String& textEncodingName)
        : ResourceResponseBase(url, mimeType, expectedLength, textEncodingName)
    {
        m_initLevel = AllFields;
    }

    WEBCORE_EXPORT void disableLazyInitialization();

    unsigned memoryUsage() const
    {
        // FIXME: Find some programmatic lighweight way to calculate ResourceResponse and associated classes.
        // This is a rough estimate of resource overhead based on stats collected from memory usage tests.
        return 3800;
        /*  1280 * 2 +                // average size of ResourceResponse. Doubled to account for the WebCore copy and the CF copy.
                                      // Mostly due to the size of the hash maps, the Header Map strings and the URL.
            256 * 2                   // Overhead from ResourceRequest, doubled to account for WebCore copy and CF copy.
                                      // Mostly due to the URL and Header Map.
         */
    }

    WEBCORE_EXPORT NSURLResponse *nsURLResponse() const;

#if USE(QUICK_LOOK)
    bool isQuickLook() const { return m_isQuickLook; }
    void setIsQuickLook(bool isQuickLook) { m_isQuickLook = isQuickLook; }
#endif

    void initNSURLResponse() const;

private:
    friend class ResourceResponseBase;

    void platformLazyInit(InitLevel);
    String platformSuggestedFilename() const;
    CertificateInfo platformCertificateInfo(std::span<const std::byte>) const;

    static bool platformCompare(const ResourceResponse& a, const ResourceResponse& b);

    mutable RetainPtr<NSURLResponse> m_nsResponse;

#if USE(QUICK_LOOK)
    bool m_isQuickLook { false };
#endif
};

} // namespace WebCore
