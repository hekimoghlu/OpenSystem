/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 15, 2023.
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

#include "ContentSecurityPolicy.h"
#include "ContentSecurityPolicyHash.h"
#include "ContentSecurityPolicyMediaListDirective.h"
#include "ContentSecurityPolicySourceListDirective.h"
#include "ContentSecurityPolicyTrustedTypesDirective.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/URL.h>

namespace WebCore {

class LocalFrame;

class ContentSecurityPolicyDirectiveList {
    WTF_MAKE_TZONE_ALLOCATED(ContentSecurityPolicyDirectiveList);
public:
    static std::unique_ptr<ContentSecurityPolicyDirectiveList> create(ContentSecurityPolicy&, const String&, ContentSecurityPolicyHeaderType, ContentSecurityPolicy::PolicyFrom);
    ContentSecurityPolicyDirectiveList(ContentSecurityPolicy&, ContentSecurityPolicyHeaderType);

    const String& header() const { return m_header; }
    ContentSecurityPolicyHeaderType headerType() const { return m_headerType; }

    const ContentSecurityPolicyDirective* violatedDirectiveForUnsafeEval() const;
    const ContentSecurityPolicyDirective* violatedDirectiveForInlineJavascriptURL(const Vector<ContentSecurityPolicyHash>&) const;
    const ContentSecurityPolicyDirective* violatedDirectiveForInlineEventHandlers(const Vector<ContentSecurityPolicyHash>&) const;
    const ContentSecurityPolicyDirective* violatedDirectiveForUnsafeInlineScriptElement(const String&, const Vector<ContentSecurityPolicyHash>&) const;
    const ContentSecurityPolicyDirective* violatedDirectiveForNonParserInsertedScripts(const String&, const Vector<ContentSecurityPolicyHash>&, const Vector<ResourceCryptographicDigest>&, const URL&, ParserInserted) const;
    const ContentSecurityPolicyDirective* violatedDirectiveForUnsafeInlineStyleElement(const String&, const Vector<ContentSecurityPolicyHash>&) const;
    const ContentSecurityPolicyDirective* violatedDirectiveForUnsafeInlineStyleAttribute(const String&, const Vector<ContentSecurityPolicyHash>&) const;

    const ContentSecurityPolicyDirective* violatedDirectiveForScriptNonce(const String&) const;
    const ContentSecurityPolicyDirective* violatedDirectiveForStyleNonce(const String&) const;

    const ContentSecurityPolicyDirective* violatedDirectiveForBaseURI(const URL&) const;
    const ContentSecurityPolicyDirective* violatedDirectiveForChildContext(const URL&, bool didReceiveRedirectResponse) const;
    const ContentSecurityPolicyDirective* violatedDirectiveForConnectSource(const URL&, bool didReceiveRedirectResponse) const;
    const ContentSecurityPolicyDirective* violatedDirectiveForFont(const URL&, bool didReceiveRedirectResponse) const;
    const ContentSecurityPolicyDirective* violatedDirectiveForFormAction(const URL&, bool didReceiveRedirectResponse) const;
    const ContentSecurityPolicyDirective* violatedDirectiveForFrame(const URL&, bool didReceiveRedirectResponse) const;
    const ContentSecurityPolicyDirective* violatedDirectiveForFrameAncestor(const LocalFrame&) const;
    const ContentSecurityPolicyDirective* violatedDirectiveForFrameAncestorOrigins(const Vector<Ref<SecurityOrigin>>&) const;
    const ContentSecurityPolicyDirective* violatedDirectiveForImage(const URL&, bool didReceiveRedirectResponse) const;
    const ContentSecurityPolicyDirective* violatedDirectiveForPrefetch(const URL&, bool didReceiveRedirectResponse) const;
#if ENABLE(APPLICATION_MANIFEST)
    const ContentSecurityPolicyDirective* violatedDirectiveForManifest(const URL&, bool didReceiveRedirectResponse) const;
#endif
    const ContentSecurityPolicyDirective* violatedDirectiveForMedia(const URL&, bool didReceiveRedirectResponse) const;
    const ContentSecurityPolicyDirective* violatedDirectiveForObjectSource(const URL&, bool didReceiveRedirectResponse, ContentSecurityPolicySourceListDirective::ShouldAllowEmptyURLIfSourceListIsNotNone) const;
    const ContentSecurityPolicyDirective* violatedDirectiveForPluginType(const String& type, const String& typeAttribute) const;
    const ContentSecurityPolicyDirective* violatedDirectiveForScript(const URL&, bool didReceiveRedirectResponse, const Vector<ResourceCryptographicDigest>&, const String&) const;
    const ContentSecurityPolicyDirective* violatedDirectiveForStyle(const URL&, bool didReceiveRedirectResponse, const String&) const;
    const ContentSecurityPolicyDirective* violatedDirectiveForWorker(const URL&, bool didReceiveRedirectResponse);
    const ContentSecurityPolicyDirective* violatedDirectiveForTrustedTypesPolicy(const String&, bool isDuplicate, AllowTrustedTypePolicy&) const;

    const ContentSecurityPolicyDirective* defaultSrc() const { return m_defaultSrc.get(); }

    bool hasBlockAllMixedContentDirective() const { return m_hasBlockAllMixedContentDirective; }
    bool hasFrameAncestorsDirective() const { return !!m_frameAncestors; }
    bool requiresTrustedTypesForScript() const { return m_requireTrustedTypesForScript; }

    const String& evalDisabledErrorMessage() const { return m_evalDisabledErrorMessage; }
    const String& webAssemblyDisabledErrorMessage() const { return m_webAssemblyDisabledErrorMessage; }
    bool isReportOnly() const { return m_reportOnly; }
    bool shouldReportSample(const String&) const;
    HashAlgorithmSet reportHash() const;
    const Vector<String>& reportToTokens() const { return m_reportToTokens; }
    const Vector<String>& reportURIs() const { return m_reportURIs; }

    // FIXME: Remove this once we teach ContentSecurityPolicyDirectiveList how to log an arbitrary console message.
    const ContentSecurityPolicy& policy() const { return m_policy; }

    bool strictDynamicIncluded() const;

private:
    void parse(const String&, ContentSecurityPolicy::PolicyFrom);

    struct ParsedDirective {
        String name;
        String value;
    };
    template<typename CharacterType> std::optional<ParsedDirective> parseDirective(std::span<const CharacterType>);
    void parseReportTo(ParsedDirective&&);
    void parseReportURI(ParsedDirective&&);
    void parseRequireTrustedTypesFor(ParsedDirective&&);
    void addDirective(ParsedDirective&&);
    void applySandboxPolicy(ParsedDirective&&);
    void setUpgradeInsecureRequests(ParsedDirective&&);
    void setBlockAllMixedContentEnabled(ParsedDirective&&);

    const ContentSecurityPolicySourceListDirective* hashReportDirectiveForScript() const;

    template <class CSPDirectiveType>
    void setCSPDirective(ParsedDirective&&, std::unique_ptr<CSPDirectiveType>&);

    ContentSecurityPolicySourceListDirective* operativeDirective(ContentSecurityPolicySourceListDirective*, const String&) const;
    ContentSecurityPolicySourceListDirective* operativeDirectiveScript(ContentSecurityPolicySourceListDirective*, const String&) const;
    ContentSecurityPolicySourceListDirective* operativeDirectiveStyle(ContentSecurityPolicySourceListDirective*, const String&) const;
    ContentSecurityPolicySourceListDirective* operativeDirectiveForWorkerSrc(ContentSecurityPolicySourceListDirective*, const String&) const;

    void setEvalDisabledErrorMessage(const String& errorMessage) { m_evalDisabledErrorMessage = errorMessage; }
    void setWebAssemblyDisabledErrorMessage(const String& errorMessage) { m_webAssemblyDisabledErrorMessage = errorMessage; }

    // FIXME: Make this a const reference once we teach applySandboxPolicy() to store its policy as opposed to applying it directly onto ContentSecurityPolicy.
    ContentSecurityPolicy& m_policy;

    String m_header;
    ContentSecurityPolicyHeaderType m_headerType;

    bool m_reportOnly { false };
    bool m_haveSandboxPolicy { false };
    bool m_upgradeInsecureRequests { false };
    bool m_hasBlockAllMixedContentDirective { false };
    bool m_requireTrustedTypesForScript { false };

    std::unique_ptr<ContentSecurityPolicyMediaListDirective> m_pluginTypes;
    std::unique_ptr<ContentSecurityPolicySourceListDirective> m_baseURI;
    std::unique_ptr<ContentSecurityPolicySourceListDirective> m_connectSrc;
    std::unique_ptr<ContentSecurityPolicySourceListDirective> m_childSrc;
    std::unique_ptr<ContentSecurityPolicySourceListDirective> m_defaultSrc;
    std::unique_ptr<ContentSecurityPolicySourceListDirective> m_fontSrc;
    std::unique_ptr<ContentSecurityPolicySourceListDirective> m_formAction;
    std::unique_ptr<ContentSecurityPolicySourceListDirective> m_frameAncestors;
    std::unique_ptr<ContentSecurityPolicySourceListDirective> m_frameSrc;
    std::unique_ptr<ContentSecurityPolicySourceListDirective> m_imgSrc;
#if ENABLE(APPLICATION_MANIFEST)
    std::unique_ptr<ContentSecurityPolicySourceListDirective> m_manifestSrc;
#endif
    std::unique_ptr<ContentSecurityPolicySourceListDirective> m_mediaSrc;
    std::unique_ptr<ContentSecurityPolicySourceListDirective> m_objectSrc;
    std::unique_ptr<ContentSecurityPolicySourceListDirective> m_prefetchSrc;
    std::unique_ptr<ContentSecurityPolicySourceListDirective> m_scriptSrc;
    std::unique_ptr<ContentSecurityPolicySourceListDirective> m_styleSrc;
    std::unique_ptr<ContentSecurityPolicySourceListDirective> m_scriptSrcElem;
    std::unique_ptr<ContentSecurityPolicySourceListDirective> m_scriptSrcAttr;
    std::unique_ptr<ContentSecurityPolicySourceListDirective> m_styleSrcElem;
    std::unique_ptr<ContentSecurityPolicySourceListDirective> m_styleSrcAttr;
    std::unique_ptr<ContentSecurityPolicyTrustedTypesDirective> m_trustedTypes;
    std::unique_ptr<ContentSecurityPolicySourceListDirective> m_workerSrc;

    Vector<String> m_reportToTokens;
    Vector<String> m_reportURIs;
    
    String m_evalDisabledErrorMessage;
    String m_webAssemblyDisabledErrorMessage;
};

} // namespace WebCore
