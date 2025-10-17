/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 13, 2022.
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

#include "CSSPropertyNames.h"
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class WillChangeData : public RefCounted<WillChangeData> {
    WTF_MAKE_TZONE_ALLOCATED(WillChangeData);
public:
    static Ref<WillChangeData> create()
    {
        return adoptRef(*new WillChangeData);
    }
    
    bool operator==(const WillChangeData&) const;

    bool isAuto() const { return m_animatableFeatures.isEmpty(); }
    size_t numFeatures() const { return m_animatableFeatures.size(); }

    bool containsScrollPosition() const;
    bool containsContents() const;
    bool containsProperty(CSSPropertyID) const;

    bool createsContainingBlockForAbsolutelyPositioned(bool isRootElement) const;
    bool createsContainingBlockForOutOfFlowPositioned(bool isRootElement) const;
    bool canCreateStackingContext() const { return m_canCreateStackingContext; }
    bool canBeBackdropRoot() const;
    bool canTriggerCompositing() const { return m_canTriggerCompositing; }
    bool canTriggerCompositingOnInline() const { return m_canTriggerCompositingOnInline; }

    enum class Feature: uint8_t {
        ScrollPosition,
        Contents,
        Property,
        Invalid
    };

    void addFeature(Feature, CSSPropertyID = CSSPropertyInvalid);
    
    typedef std::pair<Feature, CSSPropertyID> FeaturePropertyPair;
    FeaturePropertyPair featureAt(size_t) const;

    static bool propertyCreatesStackingContext(CSSPropertyID);

private:
    WillChangeData()
    {
    }

    struct AnimatableFeature {
        static const int numCSSPropertyIDBits = 14;
        static_assert(numCSSProperties < (1 << numCSSPropertyIDBits), "CSSPropertyID should fit in 14_bits");

        Feature m_feature { Feature::Property };
        unsigned m_cssPropertyID : numCSSPropertyIDBits { CSSPropertyInvalid };

        Feature feature() const
        {
            return m_feature;
        }

        CSSPropertyID property() const
        {
            return feature() == Feature::Property ? static_cast<CSSPropertyID>(m_cssPropertyID) : CSSPropertyInvalid;
        }
        
        FeaturePropertyPair featurePropertyPair() const
        {
            return FeaturePropertyPair(feature(), property());
        }

        AnimatableFeature(Feature willChange, CSSPropertyID willChangeProperty = CSSPropertyInvalid)
        {
            switch (willChange) {
            case Feature::Property:
                ASSERT(willChangeProperty != CSSPropertyInvalid);
                m_cssPropertyID = willChangeProperty;
                FALLTHROUGH;
            case Feature::ScrollPosition:
            case Feature::Contents:
                m_feature = willChange;
                break;
            case Feature::Invalid:
                ASSERT_NOT_REACHED();
                break;
            }
        }
        
        friend bool operator==(const AnimatableFeature&, const AnimatableFeature&) = default;
    };

    Vector<AnimatableFeature, 1> m_animatableFeatures;
    bool m_canCreateStackingContext { false };
    bool m_canTriggerCompositing { false };
    bool m_canTriggerCompositingOnInline { false };
};

} // namespace WebCore
