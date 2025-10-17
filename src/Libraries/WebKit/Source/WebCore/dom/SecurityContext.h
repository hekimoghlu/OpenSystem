/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 22, 2025.
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

#include "CrossOriginEmbedderPolicy.h"
#include "CrossOriginOpenerPolicy.h"
#include "SandboxFlags.h"
#include <memory>
#include <wtf/Forward.h>
#include <wtf/OptionSet.h>
#include <wtf/RefPtr.h>

namespace WebCore {

class SecurityOrigin;
class SecurityOriginPolicy;
class ContentSecurityPolicy;
struct CrossOriginOpenerPolicy;
struct PolicyContainer;
enum class ReferrerPolicy : uint8_t;

class SecurityContext {
public:
    // https://html.spec.whatwg.org/multipage/origin.html#determining-the-creation-sandboxing-flags
    SandboxFlags creationSandboxFlags() const { return m_creationSandboxFlags; }

    SandboxFlags sandboxFlags() const { return m_sandboxFlags; }
    WEBCORE_EXPORT ContentSecurityPolicy* contentSecurityPolicy();
    CheckedPtr<ContentSecurityPolicy> checkedContentSecurityPolicy();

    bool isSecureTransitionTo(const URL&) const;

    enum class SandboxFlagsSource : bool { CSP, Other };
    void enforceSandboxFlags(SandboxFlags, SandboxFlagsSource = SandboxFlagsSource::Other);

    bool isSandboxed(SandboxFlag flag) const { return m_sandboxFlags.contains(flag); }

    SecurityOriginPolicy* securityOriginPolicy() const;

    bool hasEmptySecurityOriginPolicyAndContentSecurityPolicy() const { return m_hasEmptySecurityOriginPolicy && m_hasEmptyContentSecurityPolicy; }
    bool hasInitializedSecurityOriginPolicyOrContentSecurityPolicy() const { return m_securityOriginPolicy || m_contentSecurityPolicy; }

    // Explicitly override the security origin for this security context.
    // Note: It is dangerous to change the security origin of a script context
    //       that already contains content.
    void setSecurityOriginPolicy(RefPtr<SecurityOriginPolicy>&&);

    // Explicitly override the content security policy for this security context.
    // Note: It is dangerous to change the content security policy of a script
    //       context that already contains content.
    void setContentSecurityPolicy(std::unique_ptr<ContentSecurityPolicy>&&);

    inline void setEmptySecurityOriginPolicyAndContentSecurityPolicy();

    const CrossOriginEmbedderPolicy& crossOriginEmbedderPolicy() const { return m_crossOriginEmbedderPolicy; }
    void setCrossOriginEmbedderPolicy(const CrossOriginEmbedderPolicy& crossOriginEmbedderPolicy) { m_crossOriginEmbedderPolicy = crossOriginEmbedderPolicy; }

    virtual const CrossOriginOpenerPolicy& crossOriginOpenerPolicy() const { return m_crossOriginOpenerPolicy; }
    void setCrossOriginOpenerPolicy(const CrossOriginOpenerPolicy& crossOriginOpenerPolicy) { m_crossOriginOpenerPolicy = crossOriginOpenerPolicy; }

    virtual ReferrerPolicy referrerPolicy() const { return m_referrerPolicy; }
    void setReferrerPolicy(ReferrerPolicy);

    WEBCORE_EXPORT PolicyContainer policyContainer() const;
    virtual void inheritPolicyContainerFrom(const PolicyContainer&);

    WEBCORE_EXPORT SecurityOrigin* securityOrigin() const;
    WEBCORE_EXPORT RefPtr<SecurityOrigin> protectedSecurityOrigin() const;

    static SandboxFlags parseSandboxPolicy(StringView policy, String& invalidTokensErrorMessage);
    static bool isSupportedSandboxPolicy(StringView);

    enum MixedContentType : uint8_t {
        Inactive = 1 << 0,
        Active = 1 << 1,
    };

    bool usedLegacyTLS() const { return m_usedLegacyTLS; }
    void setUsedLegacyTLS(bool used) { m_usedLegacyTLS = used; }
    const OptionSet<MixedContentType>& foundMixedContent() const { return m_mixedContentTypes; }
    bool wasPrivateRelayed() const { return m_wasPrivateRelayed; }
    void setWasPrivateRelayed(bool privateRelayed) { m_wasPrivateRelayed = privateRelayed; }
    void setFoundMixedContent(MixedContentType type) { m_mixedContentTypes.add(type); }
    bool geolocationAccessed() const { return m_geolocationAccessed; }
    void setGeolocationAccessed() { m_geolocationAccessed = true; }
    bool secureCookiesAccessed() const { return m_secureCookiesAccessed; }
    void setSecureCookiesAccessed() { m_secureCookiesAccessed = true; }

    bool isStrictMixedContentMode() const { return m_isStrictMixedContentMode; }
    void setStrictMixedContentMode(bool strictMixedContentMode) { m_isStrictMixedContentMode = strictMixedContentMode; }

    // This method implements the "Is the environment settings object settings a secure context?" algorithm from
    // the Secure Context spec: https://w3c.github.io/webappsec-secure-contexts/#settings-object (Editor's Draft, 17 November 2016)
    virtual bool isSecureContext() const = 0;

    bool haveInitializedSecurityOrigin() const { return m_haveInitializedSecurityOrigin; }

protected:
    SecurityContext();
    virtual ~SecurityContext();

    // It's only appropriate to call this during security context initialization; it's needed for
    // flags that can't be disabled with allow-* attributes, such as SandboxFlag::Navigation.
    void disableSandboxFlags(SandboxFlags flags) { m_sandboxFlags.remove(flags); }

    void didFailToInitializeSecurityOrigin() { m_haveInitializedSecurityOrigin = false; }

private:
    virtual void securityOriginDidChange() { };
    void addSandboxFlags(SandboxFlags);
    virtual std::unique_ptr<ContentSecurityPolicy> makeEmptyContentSecurityPolicy() = 0;

    RefPtr<SecurityOriginPolicy> m_securityOriginPolicy;
    std::unique_ptr<ContentSecurityPolicy> m_contentSecurityPolicy;
    CrossOriginEmbedderPolicy m_crossOriginEmbedderPolicy;
    CrossOriginOpenerPolicy m_crossOriginOpenerPolicy;
    SandboxFlags m_creationSandboxFlags;
    SandboxFlags m_sandboxFlags;
    ReferrerPolicy m_referrerPolicy { ReferrerPolicy::Default };
    OptionSet<MixedContentType> m_mixedContentTypes;
    bool m_haveInitializedSecurityOrigin { false };
    bool m_geolocationAccessed { false };
    bool m_secureCookiesAccessed { false };
    bool m_isStrictMixedContentMode { false };
    bool m_usedLegacyTLS { false };
    bool m_wasPrivateRelayed { false };
    bool m_hasEmptySecurityOriginPolicy { false };
    bool m_hasEmptyContentSecurityPolicy { false };
};

void SecurityContext::setEmptySecurityOriginPolicyAndContentSecurityPolicy()
{
    ASSERT(!m_securityOriginPolicy);
    ASSERT(!m_contentSecurityPolicy);
    m_haveInitializedSecurityOrigin = true;
    m_hasEmptySecurityOriginPolicy = true;
    m_hasEmptyContentSecurityPolicy = true;
}

} // namespace WebCore
