/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 8, 2022.
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

#include "CSSParserContext.h"
#include <optional>
#include <wtf/CheckedRef.h>
#include <wtf/Function.h>
#include <wtf/HashMap.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/URL.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/AtomStringHash.h>

namespace WebCore {

class CSSStyleSheet;
class CachedCSSStyleSheet;
class CachedResource;
class Document;
class FrameLoader;
class Node;
class SecurityOrigin;
class StyleRuleBase;
class StyleRuleImport;
class StyleRuleLayer;
class StyleRuleNamespace;

enum class CachePolicy : uint8_t;

class StyleSheetContents final : public RefCountedAndCanMakeWeakPtr<StyleSheetContents> {
public:
    static Ref<StyleSheetContents> create(const CSSParserContext& context = CSSParserContext(HTMLStandardMode))
    {
        return adoptRef(*new StyleSheetContents(nullptr, String(), context));
    }
    static Ref<StyleSheetContents> create(const String& originalURL, const CSSParserContext& context)
    {
        return adoptRef(*new StyleSheetContents(nullptr, originalURL, context));
    }
    static Ref<StyleSheetContents> create(StyleRuleImport* ownerRule, const String& originalURL, const CSSParserContext& context)
    {
        return adoptRef(*new StyleSheetContents(ownerRule, originalURL, context));
    }

    WEBCORE_EXPORT ~StyleSheetContents();
    
    const CSSParserContext& parserContext() const { return m_parserContext; }
    
    const AtomString& defaultNamespace() { return m_defaultNamespace; }
    const AtomString& namespaceURIFromPrefix(const AtomString& prefix);

    bool parseAuthorStyleSheet(const CachedCSSStyleSheet*, const SecurityOrigin*);
    WEBCORE_EXPORT bool parseString(const String&);

    bool isCacheable() const;
    bool isCacheableWithNoBaseURLDependency() const;

    bool isLoading() const;
    bool subresourcesAllowReuse(CachePolicy, FrameLoader&) const;
    WEBCORE_EXPORT bool isLoadingSubresources() const;

    void checkLoaded();
    void startLoadingDynamicSheet();

    StyleSheetContents* rootStyleSheet() const;
    Node* singleOwnerNode() const;
    Document* singleOwnerDocument() const;

    ASCIILiteral charset() const { return m_parserContext.charset; }

    bool loadCompleted() const { return m_loadCompleted; }

    bool mayDependOnBaseURL() const;

    bool traverseRules(const Function<bool(const StyleRuleBase&)>& handler) const;
    bool traverseSubresources(const Function<bool(const CachedResource&)>& handler) const;

    void setIsUserStyleSheet(bool b) { m_isUserStyleSheet = b; }
    bool isUserStyleSheet() const { return m_isUserStyleSheet; }
    void setHasSyntacticallyValidCSSHeader(bool b) { m_hasSyntacticallyValidCSSHeader = b; }
    bool hasSyntacticallyValidCSSHeader() const { return m_hasSyntacticallyValidCSSHeader; }

    void parserAddNamespace(const AtomString& prefix, const AtomString& uri);
    void parserAppendRule(Ref<StyleRuleBase>&&);
    void parserSetEncodingFromCharsetRule(const String& encoding); 
    void parserSetUsesStyleBasedEditability() { m_usesStyleBasedEditability = true; }

    void clearRules();

    String encodingFromCharsetRule() const { return m_encodingFromCharsetRule; }
    const Vector<Ref<StyleRuleLayer>>& layerRulesBeforeImportRules() const { return m_layerRulesBeforeImportRules; }
    const Vector<Ref<StyleRuleImport>>& importRules() const { return m_importRules; }
    const Vector<Ref<StyleRuleNamespace>>& namespaceRules() const { return m_namespaceRules; }
    const Vector<Ref<StyleRuleBase>>& childRules() const { return m_childRules; }

    void notifyLoadedSheet(const CachedCSSStyleSheet*);
    
    StyleSheetContents* parentStyleSheet() const;
    StyleRuleImport* ownerRule() const { return m_ownerRule; }
    void clearOwnerRule() { m_ownerRule = nullptr; }
    
    // Note that href is the URL that started the redirect chain that led to
    // this style sheet. This property probably isn't useful for much except
    // the JavaScript binding (which needs to use this value for security).
    String originalURL() const { return m_originalURL; }
    const URL& baseURL() const { return m_parserContext.baseURL; }

    bool isEmpty() const { return !ruleCount(); }
    unsigned ruleCount() const;
    StyleRuleBase* ruleAt(unsigned index) const;

    bool usesStyleBasedEditability() const { return m_usesStyleBasedEditability; }

    unsigned estimatedSizeInBytes() const;
    
    bool wrapperInsertRule(Ref<StyleRuleBase>&&, unsigned index);
    bool wrapperDeleteRule(unsigned index);

    Ref<StyleSheetContents> copy() const { return adoptRef(*new StyleSheetContents(*this)); }

    void registerClient(CSSStyleSheet*);
    void unregisterClient(CSSStyleSheet*);
    bool hasOneClient() { return m_clients.size() == 1; }
    Vector<CSSStyleSheet*> clients() const { return m_clients; }

    bool isMutable() const { return m_isMutable; }
    void setMutable() { m_isMutable = true; }

    bool hasNestingRules() const;
    void clearHasNestingRulesCache() { m_hasNestingRulesCache = { }; }

    bool isInMemoryCache() const { return m_inMemoryCacheCount; }
    void addedToMemoryCache();
    void removedFromMemoryCache();

    void shrinkToFit();

    void setAsOpaque() { m_parserContext.isContentOpaque = true; }
    bool isContentOpaque() const { return m_parserContext.isContentOpaque; }

    void setLoadErrorOccured() { m_didLoadErrorOccur = true; }

    friend class CSSStyleSheet;

    bool hasResolvedNesting() const { return m_hasResolvedNesting; }
    void setHasResolvedNesting(bool value) const { m_hasResolvedNesting = value; }

private:
    WEBCORE_EXPORT StyleSheetContents(StyleRuleImport* ownerRule, const String& originalURL, const CSSParserContext&);
    StyleSheetContents(const StyleSheetContents&);

    void clearCharsetRule();

    StyleRuleImport* m_ownerRule { nullptr };

    String m_originalURL;

    String m_encodingFromCharsetRule;
    Vector<Ref<StyleRuleLayer>> m_layerRulesBeforeImportRules;
    Vector<Ref<StyleRuleImport>> m_importRules;
    Vector<Ref<StyleRuleNamespace>> m_namespaceRules;
    Vector<Ref<StyleRuleBase>> m_childRules;
    typedef UncheckedKeyHashMap<AtomString, AtomString> PrefixNamespaceURIMap;
    PrefixNamespaceURIMap m_namespaces;
    AtomString m_defaultNamespace;

    bool m_isUserStyleSheet;
    bool m_loadCompleted { false };
    bool m_hasSyntacticallyValidCSSHeader { true };
    bool m_didLoadErrorOccur { false };
    bool m_usesStyleBasedEditability { false };
    bool m_isMutable { false };
    mutable bool m_hasResolvedNesting { false };
    mutable std::optional<bool> m_hasNestingRulesCache;
    unsigned m_inMemoryCacheCount { 0 };

    CSSParserContext m_parserContext;

    Vector<CSSStyleSheet*> m_clients;
};

} // namespace WebCore
