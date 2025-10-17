/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 8, 2025.
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

#include <optional>
#include <wtf/Forward.h>

namespace WebCore {

enum class ReferrerPolicy : uint8_t {
    EmptyString,
    NoReferrer,
    NoReferrerWhenDowngrade,
    SameOrigin,
    Origin,
    StrictOrigin,
    OriginWhenCrossOrigin,
    StrictOriginWhenCrossOrigin,
    UnsafeUrl,
    Default = StrictOriginWhenCrossOrigin
};

enum class ReferrerPolicySource : uint8_t { MetaTag, HTTPHeader, ReferrerPolicyAttribute };
std::optional<ReferrerPolicy> parseReferrerPolicy(StringView, ReferrerPolicySource);
String referrerPolicyToString(const ReferrerPolicy&);

}

namespace WTF {

template<> struct EnumTraitsForPersistence<WebCore::ReferrerPolicy> {
    using values = EnumValues<
        WebCore::ReferrerPolicy,
        WebCore::ReferrerPolicy::EmptyString,
        WebCore::ReferrerPolicy::NoReferrer,
        WebCore::ReferrerPolicy::NoReferrerWhenDowngrade,
        WebCore::ReferrerPolicy::SameOrigin,
        WebCore::ReferrerPolicy::Origin,
        WebCore::ReferrerPolicy::StrictOrigin,
        WebCore::ReferrerPolicy::OriginWhenCrossOrigin,
        WebCore::ReferrerPolicy::StrictOriginWhenCrossOrigin,
        WebCore::ReferrerPolicy::UnsafeUrl
    >;
};

}
