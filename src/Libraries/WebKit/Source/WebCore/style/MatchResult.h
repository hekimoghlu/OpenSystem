/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 30, 2024.
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

#include "PropertyAllowlist.h"
#include "RuleSet.h"
#include "SelectorChecker.h"
#include "StylePropertiesInlines.h"
#include "StyleScopeOrdinal.h"
#include <wtf/Hasher.h>

namespace WebCore::Style {

enum class FromStyleAttribute : bool { No, Yes };
enum class IsCacheable : uint8_t { No, Partially, Yes };

struct MatchedProperties {
    Ref<const StyleProperties> properties;
    uint8_t linkMatchType { SelectorChecker::MatchAll };
    PropertyAllowlist allowlistType { PropertyAllowlist::None };
    ScopeOrdinal styleScopeOrdinal { ScopeOrdinal::Element };
    FromStyleAttribute fromStyleAttribute { FromStyleAttribute::No };
    CascadeLayerPriority cascadeLayerPriority { RuleSet::cascadeLayerPriorityForUnlayered };
    IsStartingStyle isStartingStyle { IsStartingStyle::No };
    IsCacheable isCacheable { IsCacheable::Yes };
};

struct MatchResult {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;
    MatchResult(bool isForLink = false)
        : isForLink(isForLink)
    { }

    bool isForLink { false };
    bool isCompletelyNonCacheable { false };
    bool hasStartingStyle { false };
    Vector<MatchedProperties> userAgentDeclarations;
    Vector<MatchedProperties> userDeclarations;
    Vector<MatchedProperties> authorDeclarations;
    Vector<CSSPropertyID, 4> nonCacheablePropertyIds;

    bool isEmpty() const { return userAgentDeclarations.isEmpty() && userDeclarations.isEmpty() && authorDeclarations.isEmpty(); }

    friend bool operator==(const MatchResult&, const MatchResult&) = default;
    bool cacheablePropertiesEqual(const MatchResult&) const;
};

inline bool operator==(const MatchedProperties& a, const MatchedProperties& b)
{
    return a.properties.ptr() == b.properties.ptr()
        && a.linkMatchType == b.linkMatchType
        && a.allowlistType == b.allowlistType
        && a.styleScopeOrdinal == b.styleScopeOrdinal
        && a.fromStyleAttribute == b.fromStyleAttribute
        && a.cascadeLayerPriority == b.cascadeLayerPriority
        && a.isStartingStyle == b.isStartingStyle
        && a.isCacheable == b.isCacheable;
}

inline bool MatchResult::cacheablePropertiesEqual(const MatchResult& other) const
{
    if (isForLink != other.isForLink || hasStartingStyle != other.hasStartingStyle)
        return false;

    // Only author style can be non-cacheable.
    if (userAgentDeclarations != other.userAgentDeclarations)
        return false;
    if (userDeclarations != other.userDeclarations)
        return false;

    // Currently the cached style contains also the non-cacheable property values from when the entry was made
    // so we can only allow styles that override the same exact properties. Content usually animates or varies the same
    // small set of properties so this doesn't make a significant difference.
    auto nonCacheableEqual = std::ranges::equal(nonCacheablePropertyIds, other.nonCacheablePropertyIds, [](auto& idA, auto& idB) {
        // This would need to check the custom property names for equality.
        if (idA == CSSPropertyCustom || idB == CSSPropertyCustom)
            return false;
        return idA == idB;
    });
    if (!nonCacheableEqual)
        return false;

    return std::ranges::equal(authorDeclarations, other.authorDeclarations, [](auto& propertiesA, auto& propertiesB) {
        if (propertiesA.isCacheable == IsCacheable::Partially && propertiesB.isCacheable == IsCacheable::Partially)
            return true;
        return propertiesA == propertiesB;
    });
}

inline void add(Hasher& hasher, const MatchedProperties& matchedProperties)
{
    // Ignore non-cacheable properties when computing hash.
    if (matchedProperties.isCacheable == IsCacheable::Partially)
        return;
    ASSERT(matchedProperties.isCacheable == IsCacheable::Yes);

    add(hasher,
        matchedProperties.properties.ptr(),
        matchedProperties.linkMatchType,
        matchedProperties.allowlistType,
        matchedProperties.styleScopeOrdinal,
        matchedProperties.fromStyleAttribute,
        matchedProperties.cascadeLayerPriority,
        matchedProperties.isStartingStyle
    );
}

inline void add(Hasher& hasher, const MatchResult& matchResult)
{
    ASSERT(!matchResult.isCompletelyNonCacheable);
    add(hasher, matchResult.isForLink, matchResult.nonCacheablePropertyIds, matchResult.userAgentDeclarations, matchResult.userDeclarations, matchResult.authorDeclarations);
}

}
