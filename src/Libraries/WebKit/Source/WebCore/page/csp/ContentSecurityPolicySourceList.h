/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 27, 2023.
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
#include "ContentSecurityPolicySource.h"
#include <wtf/OptionSet.h>
#include <wtf/RobinHoodHashSet.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class ContentSecurityPolicy;

class ContentSecurityPolicySourceList {
public:
    ContentSecurityPolicySourceList(const ContentSecurityPolicy&, const String& directiveName);

    void parse(const String&);

    bool matches(const URL&, bool didReceiveRedirectResponse) const;
    bool matches(const Vector<ContentSecurityPolicyHash>&) const;
    bool matchesAll(const Vector<ContentSecurityPolicyHash>&) const;
    bool matches(const String& nonce) const;

    OptionSet<ContentSecurityPolicyHashAlgorithm> hashAlgorithmsUsed() const { return m_hashAlgorithmsUsed; }

    bool allowInline() const { return m_allowInline && m_hashes.isEmpty() && m_nonces.isEmpty(); }
    bool allowEval() const { return m_allowEval; }
    bool allowWasmEval() const { return m_allowWasmEval; }
    bool allowSelf() const { return m_allowSelf; }
    bool isNone() const { return m_isNone; }
    bool allowNonParserInsertedScripts() const { return m_allowNonParserInsertedScripts; }
    bool allowUnsafeHashes() const { return m_allowUnsafeHashes; }
    bool shouldReportSample() const { return m_reportSample; }

    HashAlgorithmSet reportHash() const { return m_reportHash; }

private:
    struct Host {
        StringView value;
        bool hasWildcard { false };
    };
    struct Port {
        std::optional<uint16_t> value;
        bool hasWildcard { false };
    };
    struct Source {
        StringView scheme;
        Host host;
        Port port;
        String path;
    };

    bool isProtocolAllowedByStar(const URL&) const;
    bool isValidSourceForExtensionMode(const ContentSecurityPolicySourceList::Source&);
    template<typename CharacterType> void parse(StringParsingBuffer<CharacterType>);
    template<typename CharacterType> std::optional<Source> parseSource(StringParsingBuffer<CharacterType>);
    template<typename CharacterType> StringView parseScheme(StringParsingBuffer<CharacterType>);
    template<typename CharacterType> std::optional<Host> parseHost(std::span<const CharacterType>);
    template<typename CharacterType> std::optional<Port> parsePort(std::span<const CharacterType>);
    template<typename CharacterType> String parsePath(std::span<const CharacterType>);
    template<typename CharacterType> bool parseNonceSource(StringParsingBuffer<CharacterType>);
    template<typename CharacterType> bool parseHashSource(StringParsingBuffer<CharacterType>);

    const ContentSecurityPolicy& m_policy;
    Vector<ContentSecurityPolicySource> m_list;
    MemoryCompactLookupOnlyRobinHoodHashSet<String> m_nonces;
    UncheckedKeyHashSet<ContentSecurityPolicyHash> m_hashes;
    OptionSet<ContentSecurityPolicyHashAlgorithm> m_hashAlgorithmsUsed;
    String m_directiveName;
    ContentSecurityPolicyModeForExtension m_contentSecurityPolicyModeForExtension { ContentSecurityPolicyModeForExtension::None };
    bool m_allowSelf { false };
    bool m_allowStar { false };
    bool m_allowInline { false };
    bool m_allowEval { false };
    bool m_allowWasmEval { false };
    bool m_isNone { false };
    bool m_allowNonParserInsertedScripts { false };
    bool m_allowUnsafeHashes { false };
    bool m_reportSample { false };
    HashAlgorithmSet m_reportHash { 0 };
};

} // namespace WebCore
