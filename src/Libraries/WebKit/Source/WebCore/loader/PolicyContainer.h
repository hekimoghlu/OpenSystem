/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 31, 2022.
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

#include "ContentSecurityPolicyResponseHeaders.h"
#include "CrossOriginEmbedderPolicy.h"
#include "CrossOriginOpenerPolicy.h"

namespace WebCore {

// https://html.spec.whatwg.org/multipage/origin.html#policy-container
struct PolicyContainer {
    // This acts as the CSP List. It will need to be parsed by a ContentSecurityPolicy object.
    ContentSecurityPolicyResponseHeaders contentSecurityPolicyResponseHeaders;
    CrossOriginEmbedderPolicy crossOriginEmbedderPolicy;
    CrossOriginOpenerPolicy crossOriginOpenerPolicy;
    ReferrerPolicy referrerPolicy = ReferrerPolicy::Default;

    friend bool operator==(const PolicyContainer&, const PolicyContainer&) = default;

    PolicyContainer isolatedCopy() const & { return { contentSecurityPolicyResponseHeaders.isolatedCopy(), crossOriginEmbedderPolicy.isolatedCopy(), crossOriginOpenerPolicy.isolatedCopy(), referrerPolicy }; }
    PolicyContainer isolatedCopy() && { return { WTFMove(contentSecurityPolicyResponseHeaders).isolatedCopy(), WTFMove(crossOriginEmbedderPolicy).isolatedCopy(), WTFMove(crossOriginOpenerPolicy).isolatedCopy(), referrerPolicy }; }
};

WEBCORE_EXPORT void addPolicyContainerHeaders(ResourceResponse&, const PolicyContainer&);

} // namespace WebCore
