/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 21, 2022.
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
#include "config.h"
#include "ReferrerPolicy.h"

#include "HTTPParsers.h"
#include "JSFetchReferrerPolicy.h"

namespace WebCore {

enum class ShouldParseLegacyKeywords : bool { No, Yes };

static std::optional<ReferrerPolicy> parseReferrerPolicyToken(StringView policy, ShouldParseLegacyKeywords shouldParseLegacyKeywords)
{
    // "never" / "default" / "always" are legacy keywords that we support and still defined in the HTML specification:
    // https://html.spec.whatwg.org/#meta-referrer
    if (shouldParseLegacyKeywords == ShouldParseLegacyKeywords::Yes) {
        if (equalLettersIgnoringASCIICase(policy, "never"_s))
            return ReferrerPolicy::NoReferrer;
        if (equalLettersIgnoringASCIICase(policy, "always"_s))
            return ReferrerPolicy::UnsafeUrl;
        if (equalLettersIgnoringASCIICase(policy, "default"_s))
            return ReferrerPolicy::Default;
    }

    if (equalLettersIgnoringASCIICase(policy, "no-referrer"_s))
        return ReferrerPolicy::NoReferrer;
    if (equalLettersIgnoringASCIICase(policy, "unsafe-url"_s))
        return ReferrerPolicy::UnsafeUrl;
    if (equalLettersIgnoringASCIICase(policy, "origin"_s))
        return ReferrerPolicy::Origin;
    if (equalLettersIgnoringASCIICase(policy, "origin-when-cross-origin"_s))
        return ReferrerPolicy::OriginWhenCrossOrigin;
    if (equalLettersIgnoringASCIICase(policy, "same-origin"_s))
        return ReferrerPolicy::SameOrigin;
    if (equalLettersIgnoringASCIICase(policy, "strict-origin"_s))
        return ReferrerPolicy::StrictOrigin;
    if (equalLettersIgnoringASCIICase(policy, "strict-origin-when-cross-origin"_s))
        return ReferrerPolicy::StrictOriginWhenCrossOrigin;
    if (equalLettersIgnoringASCIICase(policy, "no-referrer-when-downgrade"_s))
        return ReferrerPolicy::NoReferrerWhenDowngrade;
    if (!policy.isNull() && policy.isEmpty())
        return ReferrerPolicy::EmptyString;

    return std::nullopt;
}
    
std::optional<ReferrerPolicy> parseReferrerPolicy(StringView policyString, ReferrerPolicySource source)
{
    switch (source) {
    case ReferrerPolicySource::HTTPHeader: {
        // Implementing https://www.w3.org/TR/2017/CR-referrer-policy-20170126/#parse-referrer-policy-from-header.
        std::optional<ReferrerPolicy> result;
        for (auto tokenView : policyString.split(',')) {
            auto token = parseReferrerPolicyToken(tokenView.trim(isASCIIWhitespaceWithoutFF<UChar>), ShouldParseLegacyKeywords::No);
            if (token && token.value() != ReferrerPolicy::EmptyString)
                result = token.value();
        }
        return result;
    }
    case ReferrerPolicySource::MetaTag:
        return parseReferrerPolicyToken(policyString, ShouldParseLegacyKeywords::Yes);
    case ReferrerPolicySource::ReferrerPolicyAttribute:
        return parseReferrerPolicyToken(policyString, ShouldParseLegacyKeywords::No);
    }
    ASSERT_NOT_REACHED();
    return std::nullopt;
}

String referrerPolicyToString(const ReferrerPolicy& referrerPolicy)
{
    return convertEnumerationToString(referrerPolicy);
}

} // namespace WebCore
