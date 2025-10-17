/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 25, 2023.
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

#include "HTTPCookieAcceptPolicy.h"
#include <pal/spi/cf/CFNetworkSPI.h>

namespace WebCore {

inline HTTPCookieAcceptPolicy toHTTPCookieAcceptPolicy(NSHTTPCookieAcceptPolicy policy)
{
    switch (static_cast<NSUInteger>(policy)) {
    case NSHTTPCookieAcceptPolicyAlways:
        return HTTPCookieAcceptPolicy::AlwaysAccept;
    case NSHTTPCookieAcceptPolicyNever:
        return HTTPCookieAcceptPolicy::Never;
    case NSHTTPCookieAcceptPolicyOnlyFromMainDocumentDomain:
        return HTTPCookieAcceptPolicy::OnlyFromMainDocumentDomain;
    case NSHTTPCookieAcceptPolicyExclusivelyFromMainDocumentDomain:
        return HTTPCookieAcceptPolicy::ExclusivelyFromMainDocumentDomain;
    }

    ASSERT_NOT_REACHED();
    return HTTPCookieAcceptPolicy::Never;
}

inline NSHTTPCookieAcceptPolicy toNSHTTPCookieAcceptPolicy(HTTPCookieAcceptPolicy policy)
{
    switch (policy) {
    case HTTPCookieAcceptPolicy::AlwaysAccept:
        return NSHTTPCookieAcceptPolicyAlways;
    case HTTPCookieAcceptPolicy::Never:
        return NSHTTPCookieAcceptPolicyNever;
    case HTTPCookieAcceptPolicy::OnlyFromMainDocumentDomain:
        return NSHTTPCookieAcceptPolicyOnlyFromMainDocumentDomain;
    case HTTPCookieAcceptPolicy::ExclusivelyFromMainDocumentDomain:
        return (NSHTTPCookieAcceptPolicy)NSHTTPCookieAcceptPolicyExclusivelyFromMainDocumentDomain;
    }
    ASSERT_NOT_REACHED();
    return NSHTTPCookieAcceptPolicyAlways;
}

}
