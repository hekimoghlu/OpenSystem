/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 19, 2025.
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
#include "CSSValue.h"
#include "EventTarget.h"
#include "Length.h"
#include "TimelineRangeOffset.h"
#include "WebAnimationTime.h"
#include <wtf/BitSet.h>
#include <wtf/HashMap.h>
#include <wtf/ListHashSet.h>
#include <wtf/Markable.h>
#include <wtf/OptionSet.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/AtomString.h>
#include <wtf/text/AtomStringHash.h>

namespace WebCore {

class AnimationEventBase;
class AnimationList;
class CSSAnimation;
class CSSKeywordValue;
class CSSTransition;
class StyleOriginatedAnimation;
class WebAnimation;

struct WebAnimationsMarkableDoubleTraits {
    static bool isEmptyValue(double value)
    {
        return std::isnan(value);
    }

    static constexpr double emptyValue()
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
};

enum class AnimationImpact : uint8_t {
    RequiresRecomposite     = 1 << 0,
    ForcesStackingContext   = 1 << 1
};

enum class UseAcceleratedAction : bool { No, Yes };

enum class WebAnimationType : uint8_t { CSSAnimation, CSSTransition, WebAnimation };

using MarkableDouble = Markable<double, WebAnimationsMarkableDoubleTraits>;

using WeakStyleOriginatedAnimations = Vector<WeakPtr<StyleOriginatedAnimation, WeakPtrImplWithEventTargetData>>;
using AnimationCollection = ListHashSet<Ref<WebAnimation>>;
using AnimationEvents = Vector<Ref<AnimationEventBase>>;
using CSSAnimationCollection = ListHashSet<Ref<CSSAnimation>>;

using AnimatableCSSProperty = std::variant<CSSPropertyID, AtomString>;
using AnimatableCSSPropertyToTransitionMap = UncheckedKeyHashMap<AnimatableCSSProperty, Ref<CSSTransition>>;

enum class AcceleratedEffectProperty : uint16_t {
    Invalid = 1 << 0,
    Opacity = 1 << 1,
    Transform = 1 << 2,
    Translate = 1 << 3,
    Rotate = 1 << 4,
    Scale = 1 << 5,
    OffsetPath = 1 << 6,
    OffsetDistance = 1 << 7,
    OffsetPosition = 1 << 8,
    OffsetAnchor = 1 << 9,
    OffsetRotate = 1 << 10,
    Filter = 1 << 11,
    BackdropFilter = 1 << 12
};

constexpr OptionSet<AcceleratedEffectProperty> transformRelatedAcceleratedProperties = {
    AcceleratedEffectProperty::Transform,
    AcceleratedEffectProperty::Translate,
    AcceleratedEffectProperty::Rotate,
    AcceleratedEffectProperty::Scale,
    AcceleratedEffectProperty::OffsetAnchor,
    AcceleratedEffectProperty::OffsetDistance,
    AcceleratedEffectProperty::OffsetPath,
    AcceleratedEffectProperty::OffsetPosition,
    AcceleratedEffectProperty::OffsetRotate
};

struct CSSPropertiesBitSet {
    WTF::BitSet<numCSSProperties> m_properties { };
};

using TimelineRangeValue = std::variant<TimelineRangeOffset, RefPtr<CSSNumericValue>, RefPtr<CSSKeywordValue>, String>;

enum class Scroller : uint8_t { Nearest, Root, Self };

struct ViewTimelineInsets {
    std::optional<Length> start;
    std::optional<Length> end;
    bool operator==(const ViewTimelineInsets&) const = default;
};

} // namespace WebCore

namespace WTF {

template<> struct DefaultHash<WebCore::AnimatableCSSProperty> {
    static unsigned hash(const WebCore::AnimatableCSSProperty& key) {
        return WTF::switchOn(key,
            [] (WebCore::CSSPropertyID property) {
                return DefaultHash<WebCore::CSSPropertyID>::hash(property);
            },
            [] (const AtomString& string) {
                return DefaultHash<AtomString>::hash(string);
            }
        );
    }
    static bool equal(const WebCore::AnimatableCSSProperty& a, const WebCore::AnimatableCSSProperty& b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = true;
};

template<> struct HashTraits<WebCore::AnimatableCSSProperty> : GenericHashTraits<WebCore::AnimatableCSSProperty> {
    static const bool emptyValueIsZero = true;
    static void constructDeletedValue(WebCore::AnimatableCSSProperty& slot) {
        WebCore::CSSPropertyID property;
        HashTraits<WebCore::CSSPropertyID>::constructDeletedValue(property);
        new (NotNull, &slot) WebCore::AnimatableCSSProperty(property);
    }
    static bool isDeletedValue(const WebCore::AnimatableCSSProperty& value) {
        return WTF::switchOn(value,
            [] (WebCore::CSSPropertyID property) {
                return HashTraits<WebCore::CSSPropertyID>::isDeletedValue(property);
            },
            [] (const AtomString&) {
                return false;
            }
        );
    }
};

} // namespace WTF
