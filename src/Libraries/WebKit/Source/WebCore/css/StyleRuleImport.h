/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 15, 2021.
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

#include "CachedResourceHandle.h"
#include "CachedStyleSheetClient.h"
#include "MediaQuery.h"
#include "StyleRule.h"
#include <wtf/TypeCasts.h>

namespace WebCore {

class CachedCSSStyleSheet;
class StyleSheetContents;

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleRuleImport);
class StyleRuleImport final : public StyleRuleBase {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(StyleRuleImport);
public:
    struct SupportsCondition {
        String text;
        bool conditionMatches { true };
    };

    static Ref<StyleRuleImport> create(const String& href, MQ::MediaQueryList&&, std::optional<CascadeLayerName>&&, SupportsCondition&&);
    ~StyleRuleImport();

    Ref<StyleRuleImport> copy() const { RELEASE_ASSERT_NOT_REACHED(); }

    StyleSheetContents* parentStyleSheet() const { return m_parentStyleSheet; }
    void setParentStyleSheet(StyleSheetContents* sheet) { ASSERT(sheet); m_parentStyleSheet = sheet; }
    void clearParentStyleSheet() { m_parentStyleSheet = nullptr; }
    void cancelLoad();

    String href() const { return m_strHref; }
    StyleSheetContents* styleSheet() const { return m_styleSheet.get(); }

    bool isLoading() const;
    
    const MQ::MediaQueryList& mediaQueries() const { return m_mediaQueries; }
    void setMediaQueries(MQ::MediaQueryList&& queries) { m_mediaQueries = WTFMove(queries); }

    void requestStyleSheet();
    const CachedCSSStyleSheet* cachedCSSStyleSheet() const { return m_cachedSheet.get(); }

    const std::optional<CascadeLayerName>& cascadeLayerName() const { return m_cascadeLayerName; }
    const String& supportsText() const { return m_supportsCondition.text; }
    bool supportsMatches() const { return m_supportsCondition.conditionMatches; }

private:
    // NOTE: We put the CachedStyleSheetClient in a member instead of inheriting from it
    // to avoid adding a vptr to StyleRuleImport.
    class ImportedStyleSheetClient final : public CachedStyleSheetClient {
    public:
        ImportedStyleSheetClient(StyleRuleImport* ownerRule) : m_ownerRule(ownerRule) { }
        virtual ~ImportedStyleSheetClient() = default;
        void setCSSStyleSheet(const String& href, const URL& baseURL, ASCIILiteral charset, const CachedCSSStyleSheet* sheet) final
        {
            m_ownerRule->setCSSStyleSheet(href, baseURL, charset, sheet);
        }
    private:
        StyleRuleImport* m_ownerRule;
    };

    void setCSSStyleSheet(const String& href, const URL& baseURL, ASCIILiteral charset, const CachedCSSStyleSheet*);
    friend class ImportedStyleSheetClient;

    StyleRuleImport(const String& href, MQ::MediaQueryList&&, std::optional<CascadeLayerName>&&, SupportsCondition&&);

    StyleSheetContents* m_parentStyleSheet { nullptr };

    ImportedStyleSheetClient m_styleSheetClient;
    String m_strHref;
    MQ::MediaQueryList m_mediaQueries;
    RefPtr<StyleSheetContents> m_styleSheet;
    std::optional<CascadeLayerName> m_cascadeLayerName;
    CachedResourceHandle<CachedCSSStyleSheet> m_cachedSheet;
    bool m_loading { false };
    SupportsCondition m_supportsCondition;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::StyleRuleImport)
    static bool isType(const WebCore::StyleRuleBase& rule) { return rule.isImportRule(); }
SPECIALIZE_TYPE_TRAITS_END()
