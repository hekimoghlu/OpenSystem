/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 24, 2022.
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

#include "PropertyCascade.h"
#include "StyleBuilderState.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

struct CSSRegisteredCustomProperty;

namespace Style {

class Builder {
    WTF_MAKE_TZONE_ALLOCATED(Builder);
public:
    Builder(RenderStyle&, BuilderContext&&, const MatchResult&, CascadeLevel, OptionSet<PropertyCascade::PropertyType> = PropertyCascade::normalProperties(), const UncheckedKeyHashSet<AnimatableCSSProperty>* animatedProperties = nullptr);
    ~Builder();

    void applyAllProperties();
    void applyTopPriorityProperties();
    void applyHighPriorityProperties();
    void applyNonHighPriorityProperties();

    void applyProperty(CSSPropertyID propertyID) { applyProperties(propertyID, propertyID); }
    void applyCustomProperty(const AtomString& name);

    RefPtr<const CSSCustomPropertyValue> resolveCustomPropertyForContainerQueries(const CSSCustomPropertyValue&);

    BuilderState& state() { return m_state; }

    const UncheckedKeyHashSet<AnimatableCSSProperty> overriddenAnimatedProperties() const { return m_cascade.overriddenAnimatedProperties(); }

private:
    void applyProperties(int firstProperty, int lastProperty);
    void applyLogicalGroupProperties();
    void applyCustomProperties();
    void applyCustomPropertyImpl(const AtomString&, const PropertyCascade::Property&);

    enum CustomPropertyCycleTracking { Enabled = 0, Disabled };
    template<CustomPropertyCycleTracking trackCycles>
    void applyPropertiesImpl(int firstProperty, int lastProperty);
    void applyCascadeProperty(const PropertyCascade::Property&);
    void applyRollbackCascadeProperty(const PropertyCascade::Property&, SelectorChecker::LinkMatchMask);
    void applyProperty(CSSPropertyID, CSSValue&, SelectorChecker::LinkMatchMask, CascadeLevel);
    void applyCustomPropertyValue(const CSSCustomPropertyValue&, ApplyValueType, const CSSRegisteredCustomProperty*);

    Ref<CSSValue> resolveVariableReferences(CSSPropertyID, CSSValue&);
    RefPtr<CSSCustomPropertyValue> resolveCustomPropertyValue(CSSCustomPropertyValue&);

    void applyPageSizeDescriptor(CSSValue&);

    const PropertyCascade* ensureRollbackCascadeForRevert();
    const PropertyCascade* ensureRollbackCascadeForRevertLayer();

    using RollbackCascadeKey = std::tuple<unsigned, unsigned, unsigned>;
    RollbackCascadeKey makeRollbackCascadeKey(CascadeLevel, ScopeOrdinal = ScopeOrdinal::Element, CascadeLayerPriority = 0);

    const PropertyCascade m_cascade;
    // Rollback cascades are build on demand to resolve 'revert' and 'revert-layer' keywords.
    UncheckedKeyHashMap<RollbackCascadeKey, std::unique_ptr<const PropertyCascade>> m_rollbackCascades;

    BuilderState m_state;
};

}
}
