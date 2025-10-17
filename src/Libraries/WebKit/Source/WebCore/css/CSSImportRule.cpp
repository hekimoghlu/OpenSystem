/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 26, 2024.
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
#include "CSSImportRule.h"

#include "CSSLayerBlockRule.h"
#include "CSSMarkup.h"
#include "CSSStyleSheet.h"
#include "CachedCSSStyleSheet.h"
#include "MediaList.h"
#include "MediaQueryParser.h"
#include "StyleRuleImport.h"
#include "StyleSheetContents.h"
#include <wtf/text/StringBuilder.h>

namespace WebCore {

CSSImportRule::CSSImportRule(StyleRuleImport& importRule, CSSStyleSheet* parent)
    : CSSRule(parent)
    , m_importRule(importRule)
{
}

CSSImportRule::~CSSImportRule()
{
    if (m_styleSheetCSSOMWrapper)
        m_styleSheetCSSOMWrapper->clearOwnerRule();
    if (m_mediaCSSOMWrapper)
        m_mediaCSSOMWrapper->detachFromParent();
}

String CSSImportRule::href() const
{
    return m_importRule.get().href();
}

MediaList& CSSImportRule::media() const
{
    if (!m_mediaCSSOMWrapper)
        m_mediaCSSOMWrapper = MediaList::create(const_cast<CSSImportRule*>(this));
    return *m_mediaCSSOMWrapper;
}

String CSSImportRule::layerName() const
{
    auto name = m_importRule->cascadeLayerName();
    if (!name)
        return { };

    return stringFromCascadeLayerName(*name);
}

String CSSImportRule::supportsText() const
{
    return m_importRule->supportsText();
}

String CSSImportRule::cssTextInternal(const String& urlString) const
{
    StringBuilder builder;
    builder.append("@import "_s, serializeURL(urlString));

    if (auto layerName = this->layerName(); !layerName.isNull()) {
        if (layerName.isEmpty())
            builder.append(" layer"_s);
        else
            builder.append(" layer("_s, layerName, ')');
    }

    auto supports = supportsText();
    if (!supports.isNull())
        builder.append(" supports("_s, WTFMove(supports), ')');

    if (!mediaQueries().isEmpty()) {
        builder.append(' ');
        MQ::serialize(builder, mediaQueries());
    }

    builder.append(';');
    return builder.toString();
}

String CSSImportRule::cssText() const
{
    return cssTextInternal(m_importRule->href());
}

String CSSImportRule::cssTextWithReplacementURLs(const UncheckedKeyHashMap<String, String>& replacementURLStrings, const UncheckedKeyHashMap<RefPtr<CSSStyleSheet>, String>& replacementURLStringsForCSSStyleSheet) const
{
    if (RefPtr sheet = styleSheet()) {
        auto urlString = replacementURLStringsForCSSStyleSheet.get(sheet);
        if (!urlString.isEmpty())
            return cssTextInternal(urlString);
    }

    auto urlString = m_importRule->href();
    auto replacementURLString = replacementURLStrings.get(urlString);
    return replacementURLString.isEmpty() ? cssTextInternal(urlString) : cssTextInternal(replacementURLString);
}

CSSStyleSheet* CSSImportRule::styleSheet() const
{ 
    if (!m_importRule.get().styleSheet())
        return nullptr;

    std::optional<bool> isOriginClean;
    if (const auto* cachedSheet = m_importRule->cachedCSSStyleSheet())
        isOriginClean = cachedSheet->isCORSSameOrigin();

    if (!m_styleSheetCSSOMWrapper)
        m_styleSheetCSSOMWrapper = CSSStyleSheet::create(*m_importRule.get().styleSheet(), const_cast<CSSImportRule*>(this), isOriginClean);
    return m_styleSheetCSSOMWrapper.get(); 
}

RefPtr<CSSStyleSheet> CSSImportRule::protectedStyleSheet() const
{
    return styleSheet();
}

void CSSImportRule::reattach(StyleRuleBase&)
{
    // FIXME: Implement when enabling caching for stylesheets with import rules.
    ASSERT_NOT_REACHED();
}

const MQ::MediaQueryList& CSSImportRule::mediaQueries() const
{
    return m_importRule->mediaQueries();
}

void CSSImportRule::setMediaQueries(MQ::MediaQueryList&& queries)
{
    m_importRule->setMediaQueries(WTFMove(queries));
}

void CSSImportRule::getChildStyleSheets(UncheckedKeyHashSet<RefPtr<CSSStyleSheet>>& childStyleSheets)
{
    RefPtr sheet = styleSheet();
    if (!sheet)
        return;

    if (childStyleSheets.add(sheet).isNewEntry)
        sheet->getChildStyleSheets(childStyleSheets);
}

} // namespace WebCore
