/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 6, 2025.
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
#include "MatchedDeclarationsCache.h"

#include "CSSFontSelector.h"
#include "Document.h"
#include "DocumentInlines.h"
#include "FontCascade.h"
#include "RenderStyleInlines.h"
#include "StyleLengthResolution.h"
#include "StyleResolver.h"
#include "StyleScope.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/StringHash.h>

namespace WebCore {
namespace Style {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MatchedDeclarationsCache);

MatchedDeclarationsCache::MatchedDeclarationsCache(const Resolver& owner)
    : m_owner(owner)
    , m_sweepTimer(*this, &MatchedDeclarationsCache::sweep)
{
}

MatchedDeclarationsCache::~MatchedDeclarationsCache() = default;

void MatchedDeclarationsCache::ref() const
{
    m_owner->ref();
}

void MatchedDeclarationsCache::deref() const
{
    m_owner->deref();
}

bool MatchedDeclarationsCache::isCacheable(const Element& element, const RenderStyle& style, const RenderStyle& parentStyle)
{
    // FIXME: Writing mode and direction properties modify state when applying to document element by calling
    // Document::setWritingMode/DirectionSetOnDocumentElement. We can't skip the applying by caching.
    if (&element == element.document().documentElement())
        return false;
    // FIXME: Without the following early return we hit the final assert in
    // Element::resolvePseudoElementStyle(). Making matchedPseudoElementIds
    // PseudoElementIdentifier-aware might be a possible solution.
    if (!style.pseudoElementNameArgument().isNull())
        return false;
    // content:attr() value depends on the element it is being applied to.
    if (style.hasAttrContent() || (style.pseudoElementType() != PseudoId::None && parentStyle.hasAttrContent()))
        return false;
    if (style.zoom() != RenderStyle::initialZoom())
        return false;
    if (style.writingMode().computedWritingMode() != RenderStyle::initialWritingMode()
        || style.writingMode().computedTextDirection() != RenderStyle::initialDirection())
        return false;
    if (style.usesContainerUnits())
        return false;

    // An anchor-positioned element needs to first be resolved in order to gather
    // relevant anchor-names. Style & layout interleaving uses that information to find
    // the relevant anchors that this element will be positioned relative to. Then, the
    // anchor-positioned element will be resolved once again, this time with the anchor
    // information needed to fully resolve the element.
    if (element.document().styleScope().anchorPositionedStates().contains(element))
        return false;

    // Getting computed style after a font environment change but before full style resolution may involve styles with non-current fonts.
    // Avoid caching them.
    auto& fontSelector = element.document().fontSelector();
    if (!style.fontCascade().isCurrent(fontSelector))
        return false;
    if (!parentStyle.fontCascade().isCurrent(fontSelector))
        return false;

    if (element.hasRandomKeyMap())
        return false;

    // FIXME: counter-style: we might need to resolve cache like for fontSelector here (rdar://103018993).

    return true;
}

bool MatchedDeclarationsCache::Entry::isUsableAfterHighPriorityProperties(const RenderStyle& style) const
{
    if (style.usedZoom() != renderStyle->usedZoom())
        return false;

#if ENABLE(DARK_MODE_CSS)
    if (style.colorScheme() != renderStyle->colorScheme())
        return false;
#endif

    return Style::equalForLengthResolution(style, *renderStyle);
}

unsigned MatchedDeclarationsCache::computeHash(const MatchResult& matchResult, const StyleCustomPropertyData& inheritedCustomProperties)
{
    if (matchResult.isCompletelyNonCacheable)
        return 0;

    if (matchResult.userAgentDeclarations.isEmpty() && matchResult.userDeclarations.isEmpty()) {
        bool allNonCacheable = std::ranges::all_of(matchResult.authorDeclarations, [](auto& matchedProperties) {
            return matchedProperties.isCacheable != IsCacheable::Yes;
        });
        // No point of caching if we are not applying any properties.
        if (allNonCacheable)
            return 0;
    }
    return WTF::computeHash(matchResult, &inheritedCustomProperties);
}

const MatchedDeclarationsCache::Entry* MatchedDeclarationsCache::find(unsigned hash, const MatchResult& matchResult, const StyleCustomPropertyData& inheritedCustomProperties)
{
    if (!hash)
        return nullptr;

    auto it = m_entries.find(hash);
    if (it == m_entries.end())
        return nullptr;

    auto& entry = it->value;
    if (!matchResult.cacheablePropertiesEqual(entry.matchResult))
        return nullptr;

    if (&entry.parentRenderStyle->inheritedCustomProperties() != &inheritedCustomProperties)
        return nullptr;

    return &entry;
}

void MatchedDeclarationsCache::add(const RenderStyle& style, const RenderStyle& parentStyle, const RenderStyle* userAgentAppearanceStyle, unsigned hash, const MatchResult& matchResult)
{
    constexpr unsigned additionsBetweenSweeps = 100;
    if (++m_additionsSinceLastSweep >= additionsBetweenSweeps && !m_sweepTimer.isActive()) {
        constexpr auto sweepDelay = 1_min;
        m_sweepTimer.startOneShot(sweepDelay);
    }

    auto userAgentAppearanceStyleCopy = [&]() -> std::unique_ptr<RenderStyle> {
        if (userAgentAppearanceStyle)
            return RenderStyle::clonePtr(*userAgentAppearanceStyle);
        return { };
    };

    ASSERT(hash);
    // Note that we don't cache the original RenderStyle instance. It may be further modified.
    // The RenderStyle in the cache is really just a holder for the substructures and never used as-is.
    m_entries.add(hash, Entry { matchResult, RenderStyle::clonePtr(style), RenderStyle::clonePtr(parentStyle), userAgentAppearanceStyleCopy() });
}

void MatchedDeclarationsCache::remove(unsigned hash)
{
    m_entries.remove(hash);
}

void MatchedDeclarationsCache::invalidate()
{
    m_entries.clear();
}

void MatchedDeclarationsCache::clearEntriesAffectedByViewportUnits()
{
    Ref protectedThis { *this };

    m_entries.removeIf([](auto& keyValue) {
        return keyValue.value.renderStyle->usesViewportUnits();
    });
}

void MatchedDeclarationsCache::sweep()
{
    Ref protectedThis { *this };

    // Look for cache entries containing a style declaration with a single ref and remove them.
    // This may happen when an element attribute mutation causes it to generate a new inlineStyle()
    // or presentationalHintStyle(), potentially leaving this cache with the last ref on the old one.
    auto hasOneRef = [](auto& declarations) {
        for (auto& matchedProperties : declarations) {
            if (matchedProperties.properties->hasOneRef())
                return true;
        }
        return false;
    };

    m_entries.removeIf([&](auto& keyValue) {
        auto& matchResult = keyValue.value.matchResult;
        return hasOneRef(matchResult.userAgentDeclarations) || hasOneRef(matchResult.userDeclarations) || hasOneRef(matchResult.authorDeclarations);
    });

    m_additionsSinceLastSweep = 0;
}

}
}
