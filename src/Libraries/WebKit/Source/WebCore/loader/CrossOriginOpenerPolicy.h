/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 11, 2024.
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

#include "SandboxFlags.h"
#include <wtf/URL.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class ContentSecurityPolicy;
class LocalFrame;
class ResourceResponse;
class ScriptExecutionContext;
class SecurityOrigin;

struct NavigationRequester;
struct ReportingClient;

// https://html.spec.whatwg.org/multipage/origin.html#cross-origin-opener-policy-value
enum class CrossOriginOpenerPolicyValue : uint8_t {
    UnsafeNone,
    SameOrigin,
    SameOriginPlusCOEP,
    SameOriginAllowPopups,
    NoopenerAllowPopups
};

enum class COOPDisposition : bool { Reporting , Enforce };

// https://html.spec.whatwg.org/multipage/origin.html#cross-origin-opener-policy
struct CrossOriginOpenerPolicy {
    CrossOriginOpenerPolicyValue value { CrossOriginOpenerPolicyValue::UnsafeNone };
    CrossOriginOpenerPolicyValue reportOnlyValue { CrossOriginOpenerPolicyValue::UnsafeNone };

    String reportingEndpoint;
    String reportOnlyReportingEndpoint;

    const String& reportingEndpointForDisposition(COOPDisposition) const;
    bool hasReportingEndpoint(COOPDisposition) const;

    CrossOriginOpenerPolicy isolatedCopy() const &;
    CrossOriginOpenerPolicy isolatedCopy() &&;

    friend bool operator==(const CrossOriginOpenerPolicy&, const CrossOriginOpenerPolicy&) = default;

    void addPolicyHeadersTo(ResourceResponse&) const;
};

inline const String& CrossOriginOpenerPolicy::reportingEndpointForDisposition(COOPDisposition disposition) const
{
    return disposition == COOPDisposition::Enforce ? reportingEndpoint : reportOnlyReportingEndpoint;
}

inline bool CrossOriginOpenerPolicy::hasReportingEndpoint(COOPDisposition disposition) const
{
    return !reportingEndpointForDisposition(disposition).isEmpty();
}

// https://html.spec.whatwg.org/multipage/origin.html#coop-enforcement-result
struct CrossOriginOpenerPolicyEnforcementResult {
    WEBCORE_EXPORT static CrossOriginOpenerPolicyEnforcementResult from(const URL& currentURL, Ref<SecurityOrigin>&& currentOrigin, const CrossOriginOpenerPolicy&, std::optional<NavigationRequester>, const URL& openerURL);

    URL url;
    Ref<SecurityOrigin> currentOrigin;
    CrossOriginOpenerPolicy crossOriginOpenerPolicy;
    bool isCurrentContextNavigationSource { true };
    bool needsBrowsingContextGroupSwitch { false };
    bool needsBrowsingContextGroupSwitchDueToReportOnly { false };
};

WEBCORE_EXPORT CrossOriginOpenerPolicy obtainCrossOriginOpenerPolicy(const ResourceResponse&);
WEBCORE_EXPORT std::optional<CrossOriginOpenerPolicyEnforcementResult> doCrossOriginOpenerHandlingOfResponse(ReportingClient&, const ResourceResponse&, const std::optional<NavigationRequester>&, ContentSecurityPolicy* responseCSP, SandboxFlags effectiveSandboxFlags, const String& referrer, bool isDisplayingInitialEmptyDocument, const CrossOriginOpenerPolicyEnforcementResult& currentCoopEnforcementResult);
WEBCORE_EXPORT bool coopValuesRequireBrowsingContextGroupSwitch(bool isInitialAboutBlank, CrossOriginOpenerPolicyValue activeDocumentCOOPValue, const SecurityOrigin& activeDocumentNavigationOrigin, CrossOriginOpenerPolicyValue responseCOOPValue, const SecurityOrigin& responseOrigin);

} // namespace WebCore
