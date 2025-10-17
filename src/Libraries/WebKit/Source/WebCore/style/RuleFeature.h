/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 25, 2022.
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

#include "CSSSelector.h"
#include "CommonAtomStrings.h"
#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/text/AtomString.h>
#include <wtf/text/AtomStringHash.h>

namespace WebCore {

class StyleRule;
class StyleRuleScope;

namespace Style {

class RuleData;

// FIXME: Has* values should be separated so we could describe both the :has() argument and its position in the selector.
enum class MatchElement : uint8_t {
    Subject,
    Parent,
    Ancestor,
    DirectSibling,
    IndirectSibling,
    AnySibling,
    ParentSibling,
    AncestorSibling,
    ParentAnySibling,
    AncestorAnySibling,
    HasChild,
    HasDescendant,
    HasSibling,
    HasSiblingDescendant,
    HasAnySibling,
    HasNonSubject, // FIXME: This is a catch-all for cases where :has() is in a non-subject position.
    HasScopeBreaking, // FIXME: This is a catch-all for cases where :has() contains a scope breaking sub-selector like, like :has(:is(.x .y)).
    Host,
    HostChild
};
constexpr unsigned matchElementCount = static_cast<unsigned>(MatchElement::HostChild) + 1;

enum class IsNegation : bool { No, Yes };
enum class CanBreakScope : bool { No, Yes }; // Are we inside a logical combination pseudo-class like :is() or :not(), which if we were inside a :has(), could break out of its scope?
enum class DoesBreakScope : bool { No, Yes }; // Did we find a logical combination pseudo-class like :is() or :not() with selector combinators that do break out of a :has() scope?

// For MSVC.
#pragma pack(push, 4)
struct RuleAndSelector {
    RuleAndSelector(const RuleData&);

    RefPtr<const StyleRule> styleRule;
    uint16_t selectorIndex; // Keep in sync with RuleData's selectorIndex size.
    uint16_t selectorListIndex; // Keep in sync with RuleData's selectorListIndex size.
};

struct RuleFeature : public RuleAndSelector {
    RuleFeature(const RuleData&, MatchElement, IsNegation);

    MatchElement matchElement;
    IsNegation isNegation; // Whether the selector is in a (non-paired) :not() context.
};
static_assert(sizeof(RuleFeature) <= 16, "RuleFeature is a frequently allocated object. Keep it small.");

struct RuleFeatureWithInvalidationSelector : public RuleFeature {
    RuleFeatureWithInvalidationSelector(const RuleData&, MatchElement, IsNegation, const CSSSelector* invalidationSelector);

    const CSSSelector* invalidationSelector { nullptr };
};
#pragma pack(pop)

using PseudoClassInvalidationKey = std::tuple<unsigned, uint8_t, AtomString>;

using RuleFeatureVector = Vector<RuleFeature>;

struct RuleFeatureSet {
    void add(const RuleFeatureSet&);
    void clear();
    void shrinkToFit();
    void collectFeatures(const RuleData&, const Vector<Ref<const StyleRuleScope>>& scopeRules = { });
    void registerContentAttribute(const AtomString&);

    bool usesHasPseudoClass() const;
    bool usesMatchElement(MatchElement matchElement) const { return usedMatchElements[enumToUnderlyingType(matchElement)]; }
    void setUsesMatchElement(MatchElement matchElement) { usedMatchElements[enumToUnderlyingType(matchElement)] = true; }

    UncheckedKeyHashSet<AtomString> idsInRules;
    UncheckedKeyHashSet<AtomString> idsMatchingAncestorsInRules;
    UncheckedKeyHashSet<AtomString> attributeLowercaseLocalNamesInRules;
    UncheckedKeyHashSet<AtomString> attributeLocalNamesInRules;
    UncheckedKeyHashSet<AtomString> contentAttributeNamesInRules;
    Vector<RuleAndSelector> siblingRules;
    Vector<RuleAndSelector> uncommonAttributeRules;

    UncheckedKeyHashMap<AtomString, std::unique_ptr<RuleFeatureVector>> idRules;
    UncheckedKeyHashMap<AtomString, std::unique_ptr<RuleFeatureVector>> classRules;
    UncheckedKeyHashMap<AtomString, std::unique_ptr<Vector<RuleFeatureWithInvalidationSelector>>> attributeRules;
    UncheckedKeyHashMap<PseudoClassInvalidationKey, std::unique_ptr<RuleFeatureVector>> pseudoClassRules;
    UncheckedKeyHashMap<PseudoClassInvalidationKey, std::unique_ptr<Vector<RuleFeatureWithInvalidationSelector>>> hasPseudoClassRules;
    Vector<RuleAndSelector> scopeBreakingHasPseudoClassRules;

    UncheckedKeyHashSet<AtomString> classesAffectingHost;
    UncheckedKeyHashSet<AtomString> attributesAffectingHost;
    UncheckedKeyHashSet<CSSSelector::PseudoClass, IntHash<CSSSelector::PseudoClass>, WTF::StrongEnumHashTraits<CSSSelector::PseudoClass>> pseudoClassesAffectingHost;
    UncheckedKeyHashSet<CSSSelector::PseudoClass, IntHash<CSSSelector::PseudoClass>, WTF::StrongEnumHashTraits<CSSSelector::PseudoClass>> pseudoClasses;

    std::array<bool, matchElementCount> usedMatchElements { };

    bool usesFirstLineRules { false };
    bool usesFirstLetterRules { false };
    bool hasStartingStyleRules { false };

private:
    struct SelectorFeatures {
        bool hasSiblingSelector { false };

        using InvalidationFeature = std::tuple<const CSSSelector*, MatchElement, IsNegation>;
        using HasInvalidationFeature = std::tuple<const CSSSelector*, MatchElement, IsNegation, DoesBreakScope>;

        Vector<InvalidationFeature> ids;
        Vector<InvalidationFeature> classes;
        Vector<InvalidationFeature> attributes;
        Vector<InvalidationFeature> pseudoClasses;
        Vector<HasInvalidationFeature> hasPseudoClasses;
    };
    DoesBreakScope recursivelyCollectFeaturesFromSelector(SelectorFeatures&, const CSSSelector&, MatchElement = MatchElement::Subject, IsNegation = IsNegation::No, CanBreakScope = CanBreakScope::No);
};

bool isHasPseudoClassMatchElement(MatchElement);
MatchElement computeHasPseudoClassMatchElement(const CSSSelector&);

enum class InvalidationKeyType : uint8_t { Universal = 1, Class, Id, Tag };
PseudoClassInvalidationKey makePseudoClassInvalidationKey(CSSSelector::PseudoClass, InvalidationKeyType, const AtomString& = starAtom());

inline bool isUniversalInvalidation(const PseudoClassInvalidationKey& key)
{
    return static_cast<InvalidationKeyType>(std::get<1>(key)) == InvalidationKeyType::Universal;
}

inline bool RuleFeatureSet::usesHasPseudoClass() const
{
    return usesMatchElement(MatchElement::HasChild)
        || usesMatchElement(MatchElement::HasDescendant)
        || usesMatchElement(MatchElement::HasSibling)
        || usesMatchElement(MatchElement::HasSiblingDescendant)
        || usesMatchElement(MatchElement::HasAnySibling)
        || usesMatchElement(MatchElement::HasNonSubject)
        || usesMatchElement(MatchElement::HasScopeBreaking);
}

} // namespace Style
} // namespace WebCore
