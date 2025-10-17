/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 11, 2024.
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

#include "CascadeLevel.h"
#include "MatchResult.h"
#include "WebAnimationTypes.h"
#include <wtf/BitSet.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class StyleResolver;

namespace Style {

class PropertyCascade {
    WTF_MAKE_TZONE_ALLOCATED(PropertyCascade);
public:
    using PropertyBitSet = WTF::BitSet<lastLowPriorityProperty + 1>;

    enum class PropertyType : uint8_t {
        NonInherited = 1 << 0,
        Inherited = 1 << 1,
        ExplicitlyInherited = 1 << 2,
        AfterAnimation = 1 << 3,
        AfterTransition = 1 << 4,
        StartingStyle = 1 << 5,
        NonCacheable = 1 << 6,
    };
    static constexpr OptionSet<PropertyType> normalProperties() { return { PropertyType::NonInherited,  PropertyType::Inherited }; }
    static constexpr OptionSet<PropertyType> startingStyleProperties() { return normalProperties() | PropertyType::StartingStyle; }

    PropertyCascade(const MatchResult&, CascadeLevel, OptionSet<PropertyType> includedProperties, const UncheckedKeyHashSet<AnimatableCSSProperty>* = nullptr);
    PropertyCascade(const PropertyCascade&, CascadeLevel, std::optional<ScopeOrdinal> rollbackScope = { }, std::optional<CascadeLayerPriority> maximumCascadeLayerPriorityForRollback = { });

    ~PropertyCascade();

    struct Property {
        CSSPropertyID id;
        CascadeLevel cascadeLevel;
        ScopeOrdinal styleScopeOrdinal;
        CascadeLayerPriority cascadeLayerPriority;
        FromStyleAttribute fromStyleAttribute;
        std::array<CSSValue*, 3> cssValue; // Values for link match states MatchDefault, MatchLink and MatchVisited
        std::array<CascadeLevel, 3> cascadeLevels;
    };

    bool isEmpty() const { return m_propertyIsPresent.isEmpty() && !m_seenLogicalGroupPropertyCount; }

    bool hasNormalProperty(CSSPropertyID) const;
    const Property& normalProperty(CSSPropertyID) const;

    bool hasLogicalGroupProperty(CSSPropertyID) const;
    const Property& logicalGroupProperty(CSSPropertyID) const;
    const Property* lastPropertyResolvingLogicalPropertyPair(CSSPropertyID, WritingMode) const;

    bool hasCustomProperty(const AtomString&) const;
    const Property& customProperty(const AtomString&) const;

    std::span<const CSSPropertyID> logicalGroupPropertyIDs() const;
    const UncheckedKeyHashMap<AtomString, Property>& customProperties() const { return m_customProperties; }

    const UncheckedKeyHashSet<AnimatableCSSProperty> overriddenAnimatedProperties() const;

    PropertyBitSet& propertyIsPresent() { return m_propertyIsPresent; }
    const PropertyBitSet& propertyIsPresent() const { return m_propertyIsPresent; }

private:
    void buildCascade();
    bool addNormalMatches(CascadeLevel);
    void addImportantMatches(CascadeLevel);
    bool addMatch(const MatchedProperties&, CascadeLevel, IsImportant);
    bool shouldApplyAfterAnimation(const StyleProperties::PropertyReference&);

    void set(CSSPropertyID, CSSValue&, const MatchedProperties&, CascadeLevel);
    void setLogicalGroupProperty(CSSPropertyID, CSSValue&, const MatchedProperties&, CascadeLevel);
    static void setPropertyInternal(Property&, CSSPropertyID, CSSValue&, const MatchedProperties&, CascadeLevel);

    bool hasProperty(CSSPropertyID, const CSSValue&);

    unsigned logicalGroupPropertyIndex(CSSPropertyID) const;
    void setLogicalGroupPropertyIndex(CSSPropertyID, unsigned);
    void sortLogicalGroupPropertyIDs();

    const MatchResult& m_matchResult;
    const OptionSet<PropertyType> m_includedProperties;
    const CascadeLevel m_maximumCascadeLevel;
    const std::optional<ScopeOrdinal> m_rollbackScope;
    const std::optional<CascadeLayerPriority> m_maximumCascadeLayerPriorityForRollback;

    struct AnimationLayer {
        AnimationLayer(const UncheckedKeyHashSet<AnimatableCSSProperty>&);

        const UncheckedKeyHashSet<AnimatableCSSProperty>& properties;
        UncheckedKeyHashSet<AnimatableCSSProperty> overriddenProperties;
        bool hasCustomProperties { false };
        bool hasFontSize { false };
        bool hasLineHeight { false };
    };
    std::optional<AnimationLayer> m_animationLayer;

    // The CSSPropertyID enum is sorted like this:
    // 1. CSSPropertyInvalid and CSSPropertyCustom.
    // 2. Normal longhand properties (high priority ones followed by low priority ones).
    // 3. Longhand properties in a logical property group.
    // 4. Shorthand properties.
    //
    // 'm_properties' is used for both normal and logical longhands, so it has size 'lastLogicalGroupProperty + 1'.
    // It could actually be 2 units smaller, but then we would have to subtract 'firstCSSProperty', which may not be worth it.
    // 'm_propertyIsPresent' is not used for logical group properties, so we only need to cover up to the last low priority one.
    std::array<Property, lastLogicalGroupProperty + 1> m_properties;
    PropertyBitSet m_propertyIsPresent;

    static constexpr unsigned logicalGroupPropertyCount = lastLogicalGroupProperty - firstLogicalGroupProperty + 1;
    std::array<unsigned, logicalGroupPropertyCount> m_logicalGroupPropertyIndices { };
    unsigned m_lastIndexForLogicalGroup { 0 };
    std::array<CSSPropertyID, logicalGroupPropertyCount> m_logicalGroupPropertyIDs { };
    unsigned m_seenLogicalGroupPropertyCount { 0 };
    CSSPropertyID m_lowestSeenLogicalGroupProperty { lastLogicalGroupProperty };
    CSSPropertyID m_highestSeenLogicalGroupProperty { firstLogicalGroupProperty };

    UncheckedKeyHashMap<AtomString, Property> m_customProperties;
};

inline bool PropertyCascade::hasNormalProperty(CSSPropertyID id) const
{
    ASSERT(id < firstLogicalGroupProperty);
    return m_propertyIsPresent.get(id);
}

inline const PropertyCascade::Property& PropertyCascade::normalProperty(CSSPropertyID id) const
{
    ASSERT(hasNormalProperty(id));
    return m_properties[id];
}

inline unsigned PropertyCascade::logicalGroupPropertyIndex(CSSPropertyID id) const
{
    ASSERT(id >= firstLogicalGroupProperty);
    ASSERT(id <= lastLogicalGroupProperty);
    return m_logicalGroupPropertyIndices[id - firstLogicalGroupProperty];
}

inline void PropertyCascade::setLogicalGroupPropertyIndex(CSSPropertyID id, unsigned index)
{
    ASSERT(id >= firstLogicalGroupProperty);
    ASSERT(id <= lastLogicalGroupProperty);
    m_logicalGroupPropertyIndices[id - firstLogicalGroupProperty] = index;
}

inline bool PropertyCascade::hasLogicalGroupProperty(CSSPropertyID id) const
{
    return logicalGroupPropertyIndex(id);
}

inline const PropertyCascade::Property& PropertyCascade::logicalGroupProperty(CSSPropertyID id) const
{
    ASSERT(hasLogicalGroupProperty(id));
    return m_properties[id];
}

inline std::span<const CSSPropertyID> PropertyCascade::logicalGroupPropertyIDs() const
{
    return std::span { m_logicalGroupPropertyIDs }.first(m_seenLogicalGroupPropertyCount);
}

inline bool PropertyCascade::hasCustomProperty(const AtomString& name) const
{
    return m_customProperties.contains(name);
}

inline const PropertyCascade::Property& PropertyCascade::customProperty(const AtomString& name) const
{
    ASSERT(hasCustomProperty(name));
    return m_customProperties.find(name)->value;
}

} // namespace Style
} // namespace WebCore
