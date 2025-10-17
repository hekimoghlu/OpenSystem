/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 17, 2023.
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
#include "SecurityContext.h"

#include "ContentSecurityPolicy.h"
#include "PolicyContainer.h"
#include "SecurityOrigin.h"
#include "SecurityOriginPolicy.h"
#include <wtf/text/StringBuilder.h>

namespace WebCore {

SecurityContext::SecurityContext() = default;

SecurityContext::~SecurityContext() = default;

void SecurityContext::setSecurityOriginPolicy(RefPtr<SecurityOriginPolicy>&& securityOriginPolicy)
{
    auto currentOrigin = securityOrigin() ? securityOrigin()->data() : SecurityOriginData { };
    bool haveInitializedSecurityOrigin = std::exchange(m_haveInitializedSecurityOrigin, true);

    m_securityOriginPolicy = WTFMove(securityOriginPolicy);
    m_hasEmptySecurityOriginPolicy = false;

    auto origin = securityOrigin() ? securityOrigin()->data() : SecurityOriginData { };
    if (!haveInitializedSecurityOrigin || currentOrigin != origin)
        securityOriginDidChange();
}

ContentSecurityPolicy* SecurityContext::contentSecurityPolicy()
{
    if (!m_contentSecurityPolicy && m_hasEmptyContentSecurityPolicy)
        m_contentSecurityPolicy = makeEmptyContentSecurityPolicy();
    return m_contentSecurityPolicy.get();
}

SecurityOrigin* SecurityContext::securityOrigin() const
{
    RefPtr policy = securityOriginPolicy();
    if (!policy)
        return nullptr;
    return &policy->origin();
}

RefPtr<SecurityOrigin> SecurityContext::protectedSecurityOrigin() const
{
    return securityOrigin();
}

SecurityOriginPolicy* SecurityContext::securityOriginPolicy() const
{
    if (!m_securityOriginPolicy && m_hasEmptySecurityOriginPolicy)
        const_cast<SecurityContext&>(*this).m_securityOriginPolicy = SecurityOriginPolicy::create(SecurityOrigin::createOpaque());
    return m_securityOriginPolicy.get();
}

void SecurityContext::setContentSecurityPolicy(std::unique_ptr<ContentSecurityPolicy>&& contentSecurityPolicy)
{
    m_contentSecurityPolicy = WTFMove(contentSecurityPolicy);
    m_hasEmptyContentSecurityPolicy = false;
}

bool SecurityContext::isSecureTransitionTo(const URL& url) const
{
    // If we haven't initialized our security origin by now, this is probably
    // a new window created via the API (i.e., that lacks an origin and lacks
    // a place to inherit the origin from).
    if (!haveInitializedSecurityOrigin())
        return true;

    return securityOriginPolicy()->origin().isSameOriginDomain(SecurityOrigin::create(url).get());
}

void SecurityContext::enforceSandboxFlags(SandboxFlags flags, SandboxFlagsSource source)
{
    if (source != SandboxFlagsSource::CSP)
        m_creationSandboxFlags.add(flags);
    m_sandboxFlags.add(flags);

    // The SandboxFlag::Origin is stored redundantly in the security origin.
    if (isSandboxed(SandboxFlag::Origin) && securityOriginPolicy() && !securityOriginPolicy()->origin().isOpaque())
        setSecurityOriginPolicy(SecurityOriginPolicy::create(SecurityOrigin::createOpaque()));
}

bool SecurityContext::isSupportedSandboxPolicy(StringView policy)
{
    static constexpr ASCIILiteral supportedPolicies[] = {
        "allow-top-navigation-to-custom-protocols"_s, "allow-forms"_s, "allow-same-origin"_s, "allow-scripts"_s,
        "allow-top-navigation"_s, "allow-pointer-lock"_s, "allow-popups"_s, "allow-popups-to-escape-sandbox"_s,
        "allow-top-navigation-by-user-activation"_s, "allow-modals"_s, "allow-storage-access-by-user-activation"_s,
        "allow-downloads"_s
    };

    for (auto supportedPolicy : supportedPolicies) {
        if (equalIgnoringASCIICase(policy, supportedPolicy))
            return true;
    }
    return false;
}

// Keep SecurityContext::isSupportedSandboxPolicy() in sync when updating this function.
SandboxFlags SecurityContext::parseSandboxPolicy(StringView policy, String& invalidTokensErrorMessage)
{
    // http://www.w3.org/TR/html5/the-iframe-element.html#attr-iframe-sandbox
    // Parse the unordered set of unique space-separated tokens.
    SandboxFlags flags = SandboxFlags::all();
    unsigned length = policy.length();
    unsigned start = 0;
    unsigned numberOfTokenErrors = 0;
    StringBuilder tokenErrors;
    while (true) {
        while (start < length && isASCIIWhitespace(policy[start]))
            ++start;
        if (start >= length)
            break;
        unsigned end = start + 1;
        while (end < length && !isASCIIWhitespace(policy[end]))
            ++end;

        // Turn off the corresponding sandbox flag if it's set as "allowed".
        auto sandboxToken = policy.substring(start, end - start);
        if (equalLettersIgnoringASCIICase(sandboxToken, "allow-same-origin"_s))
            flags.remove(SandboxFlag::Origin);
        else if (equalLettersIgnoringASCIICase(sandboxToken, "allow-downloads"_s))
            flags.remove(SandboxFlag::Downloads);
        else if (equalLettersIgnoringASCIICase(sandboxToken, "allow-forms"_s))
            flags.remove(SandboxFlag::Forms);
        else if (equalLettersIgnoringASCIICase(sandboxToken, "allow-scripts"_s)) {
            flags.remove(SandboxFlag::Scripts);
            flags.remove(SandboxFlag::AutomaticFeatures);
        } else if (equalLettersIgnoringASCIICase(sandboxToken, "allow-top-navigation"_s)) {
            flags.remove(SandboxFlag::TopNavigation);
            flags.remove(SandboxFlag::TopNavigationByUserActivation);
        } else if (equalLettersIgnoringASCIICase(sandboxToken, "allow-popups"_s))
            flags.remove(SandboxFlag::Popups);
        else if (equalLettersIgnoringASCIICase(sandboxToken, "allow-pointer-lock"_s))
            flags.remove(SandboxFlag::PointerLock);
        else if (equalLettersIgnoringASCIICase(sandboxToken, "allow-popups-to-escape-sandbox"_s))
            flags.remove(SandboxFlag::PropagatesToAuxiliaryBrowsingContexts);
        else if (equalLettersIgnoringASCIICase(sandboxToken, "allow-top-navigation-by-user-activation"_s))
            flags.remove(SandboxFlag::TopNavigationByUserActivation);
        else if (equalLettersIgnoringASCIICase(sandboxToken, "allow-top-navigation-to-custom-protocols"_s))
            flags.remove(SandboxFlag::TopNavigationToCustomProtocols);
        else if (equalLettersIgnoringASCIICase(sandboxToken, "allow-modals"_s))
            flags.remove(SandboxFlag::Modals);
        else if (equalLettersIgnoringASCIICase(sandboxToken, "allow-storage-access-by-user-activation"_s))
            flags.remove(SandboxFlag::StorageAccessByUserActivation);
        else {
            if (numberOfTokenErrors)
                tokenErrors.append(", '"_s);
            else
                tokenErrors.append('\'');
            tokenErrors.append(sandboxToken, '\'');
            numberOfTokenErrors++;
        }

        start = end + 1;
    }

    if (numberOfTokenErrors) {
        if (numberOfTokenErrors > 1)
            tokenErrors.append(" are invalid sandbox flags."_s);
        else
            tokenErrors.append(" is an invalid sandbox flag."_s);
        invalidTokensErrorMessage = tokenErrors.toString();
    }

    return flags;
}

void SecurityContext::setReferrerPolicy(ReferrerPolicy referrerPolicy)
{
    // Do not override existing referrer policy with the "empty string" one as the "empty string" means we should use
    // the policy defined elsewhere.
    if (referrerPolicy == ReferrerPolicy::EmptyString)
        return;

    m_referrerPolicy = referrerPolicy;
}

PolicyContainer SecurityContext::policyContainer() const
{
    ASSERT(m_contentSecurityPolicy);
    return {
        m_contentSecurityPolicy->responseHeaders(),
        crossOriginEmbedderPolicy(),
        crossOriginOpenerPolicy(),
        referrerPolicy()
    };
}

void SecurityContext::inheritPolicyContainerFrom(const PolicyContainer& policyContainer)
{
    if (!contentSecurityPolicy())
        setContentSecurityPolicy(makeUnique<ContentSecurityPolicy>(URL { }, nullptr, nullptr));

    checkedContentSecurityPolicy()->inheritHeadersFrom(policyContainer.contentSecurityPolicyResponseHeaders);
    setCrossOriginOpenerPolicy(policyContainer.crossOriginOpenerPolicy);
    setCrossOriginEmbedderPolicy(policyContainer.crossOriginEmbedderPolicy);
    setReferrerPolicy(policyContainer.referrerPolicy);
}

CheckedPtr<ContentSecurityPolicy> SecurityContext::checkedContentSecurityPolicy()
{
    return contentSecurityPolicy();
}

}
