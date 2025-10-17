/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 28, 2023.
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

#include "HTTPHeaderMap.h"
#include "ResourceLoadPriority.h"
#include "ResourceRequestBase.h"
#include <pal/spi/cf/CFNetworkSPI.h>

namespace WebCore {

class ResourceRequest;

#if HAVE(CFNETWORK_NSURLSESSION_CONNECTION_CACHE_LIMITS)
inline ResourceLoadPriority toResourceLoadPriority(CFURLRequestPriority priority)
{
    switch (priority) {
    case -1:
    case 0:
        return ResourceLoadPriority::VeryLow;
    case 1:
        return ResourceLoadPriority::Low;
    case 2:
        return ResourceLoadPriority::Medium;
    case 3:
        return ResourceLoadPriority::High;
    case 4:
        return ResourceLoadPriority::VeryHigh;
    default:
        ASSERT_NOT_REACHED();
        return ResourceLoadPriority::Lowest;
    }
}

inline CFURLRequestPriority toPlatformRequestPriority(ResourceLoadPriority priority)
{
    switch (priority) {
    case ResourceLoadPriority::VeryLow:
        return 0;
    case ResourceLoadPriority::Low:
        return 1;
    case ResourceLoadPriority::Medium:
        return 2;
    case ResourceLoadPriority::High:
        return 3;
    case ResourceLoadPriority::VeryHigh:
        return 4;
    }

    ASSERT_NOT_REACHED();
    return 0;
}

#else

inline ResourceLoadPriority toResourceLoadPriority(CFURLRequestPriority priority)
{
    switch (priority) {
    case -1:
        return ResourceLoadPriority::VeryLow;
    case 0:
        return ResourceLoadPriority::Low;
    case 1:
        return ResourceLoadPriority::Medium;
    case 2:
        return ResourceLoadPriority::High;
    case 3:
        return ResourceLoadPriority::VeryHigh;
    default:
        ASSERT_NOT_REACHED();
        return ResourceLoadPriority::Lowest;
    }
}

inline CFURLRequestPriority toPlatformRequestPriority(ResourceLoadPriority priority)
{
    switch (priority) {
    case ResourceLoadPriority::VeryLow:
        return -1;
    case ResourceLoadPriority::Low:
        return 0;
    case ResourceLoadPriority::Medium:
        return 1;
    case ResourceLoadPriority::High:
        return 2;
    case ResourceLoadPriority::VeryHigh:
        return 3;
    }

    ASSERT_NOT_REACHED();
    return 0;
}
#endif


inline RetainPtr<CFStringRef> httpHeaderValueUsingSuitableEncoding(HTTPHeaderMap::const_iterator::KeyValue header)
{
    if (header.keyAsHTTPHeaderName && *header.keyAsHTTPHeaderName == HTTPHeaderName::LastEventID && !header.value.containsOnlyASCII()) {
        auto utf8Value = header.value.utf8();
        auto utf8ValueSpan = utf8Value.span();
        // Constructing a string with the UTF-8 bytes but claiming that itâ€™s Latin-1 is the way to get CFNetwork to put those UTF-8 bytes on the wire.
        return adoptCF(CFStringCreateWithBytes(nullptr, utf8ValueSpan.data(), utf8ValueSpan.size(), kCFStringEncodingISOLatin1, false));
    }
    return header.value.createCFString();
}

} // namespace WebCore
