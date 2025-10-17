/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 2, 2023.
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

#include "CompositeOperation.h"
#include "KeyframeInterpolation.h"
#include "RenderStyle.h"
#include "WebAnimationTypes.h"
#include <wtf/Vector.h>
#include <wtf/HashSet.h>
#include <wtf/text/AtomString.h>
#include <wtf/text/AtomStringHash.h>

namespace WebCore {

class KeyframeEffect;
class StyleProperties;
class TimingFunction;

namespace Style {
class Resolver;
}

class BlendingKeyframe final : public KeyframeInterpolation::Keyframe {
public:
    BlendingKeyframe(double offset, std::unique_ptr<RenderStyle> style)
        : m_offset(offset)
        , m_style(WTFMove(style))
    {
    }

    // KeyframeInterpolation::Keyframe
    double offset() const final { return m_offset; }
    std::optional<CompositeOperation> compositeOperation() const final { return m_compositeOperation; }
    bool animatesProperty(KeyframeInterpolation::Property) const final;
    bool isBlendingKeyframe() const final { return true; }

    void addProperty(const AnimatableCSSProperty&);
    const UncheckedKeyHashSet<AnimatableCSSProperty>& properties() const { return m_properties; }

    void setOffset(double offset) { m_offset = offset; }

    const RenderStyle* style() const { return m_style.get(); }
    void setStyle(std::unique_ptr<RenderStyle> style) { m_style = WTFMove(style); }

    TimingFunction* timingFunction() const { return m_timingFunction.get(); }
    void setTimingFunction(const RefPtr<TimingFunction>& timingFunction) { m_timingFunction = timingFunction; }

    void setCompositeOperation(std::optional<CompositeOperation> op) { m_compositeOperation = op; }

    bool containsDirectionAwareProperty() const { return m_containsDirectionAwareProperty; }
    void setContainsDirectionAwareProperty(bool containsDirectionAwareProperty) { m_containsDirectionAwareProperty = containsDirectionAwareProperty; }

private:
    double m_offset;
    UncheckedKeyHashSet<AnimatableCSSProperty> m_properties; // The properties specified in this keyframe.
    std::unique_ptr<RenderStyle> m_style;
    RefPtr<TimingFunction> m_timingFunction;
    std::optional<CompositeOperation> m_compositeOperation;
    bool m_containsDirectionAwareProperty { false };
};

class BlendingKeyframes {
public:
    explicit BlendingKeyframes(const AtomString& animationName)
        : m_animationName(animationName)
    {
    }
    ~BlendingKeyframes();

    BlendingKeyframes& operator=(BlendingKeyframes&&) = default;
    bool operator==(const BlendingKeyframes&) const;

    const AtomString& animationName() const { return m_animationName; }

    void insert(BlendingKeyframe&&);

    void addProperty(const AnimatableCSSProperty&);
    bool containsProperty(const AnimatableCSSProperty&) const;
    const UncheckedKeyHashSet<AnimatableCSSProperty>& properties() const { return m_properties; }

    bool containsAnimatableCSSProperty() const;
    bool containsDirectionAwareProperty() const;

    void clear();
    bool isEmpty() const { return m_keyframes.isEmpty(); }
    size_t size() const { return m_keyframes.size(); }
    const BlendingKeyframe& operator[](size_t index) const { return m_keyframes[index]; }

    void copyKeyframes(const BlendingKeyframes&);
    bool hasImplicitKeyframes() const;
    bool hasImplicitKeyframeForProperty(AnimatableCSSProperty) const;
    void fillImplicitKeyframes(const KeyframeEffect&, const RenderStyle& elementStyle);

    auto begin() const { return m_keyframes.begin(); }
    auto end() const { return m_keyframes.end(); }

    bool usesContainerUnits() const;
    bool usesRelativeFontWeight() const;
    bool hasCSSVariableReferences() const;
    bool hasColorSetToCurrentColor() const;
    bool hasPropertySetToCurrentColor() const;
    const UncheckedKeyHashSet<AnimatableCSSProperty>& propertiesSetToInherit() const;

    void updatePropertiesMetadata(const StyleProperties&);

    bool hasWidthDependentTransform() const { return m_hasWidthDependentTransform; }
    bool hasHeightDependentTransform() const { return m_hasHeightDependentTransform; }
    bool hasDiscreteTransformInterval() const { return m_hasDiscreteTransformInterval; }
    bool hasExplicitlyInheritedKeyframeProperty() const { return m_hasExplicitlyInheritedKeyframeProperty; }
    bool usesAnchorFunctions() const { return m_usesAnchorFunctions; }

private:
    void analyzeKeyframe(const BlendingKeyframe&);

    AtomString m_animationName;
    Vector<BlendingKeyframe> m_keyframes; // Kept sorted by key.
    UncheckedKeyHashSet<AnimatableCSSProperty> m_properties; // The properties being animated.
    UncheckedKeyHashSet<AnimatableCSSProperty> m_explicitToProperties; // The properties with an explicit value for the 100% keyframe.
    UncheckedKeyHashSet<AnimatableCSSProperty> m_explicitFromProperties; // The properties with an explicit value for the 0% keyframe.
    UncheckedKeyHashSet<AnimatableCSSProperty> m_propertiesSetToInherit;
    UncheckedKeyHashSet<AnimatableCSSProperty> m_propertiesSetToCurrentColor;
    bool m_usesRelativeFontWeight { false };
    bool m_containsCSSVariableReferences { false };
    bool m_usesAnchorFunctions { false };
    bool m_hasWidthDependentTransform { false };
    bool m_hasHeightDependentTransform { false };
    bool m_hasDiscreteTransformInterval { false };
    bool m_hasExplicitlyInheritedKeyframeProperty { false };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_KEYFRAME_INTERPOLATION_KEYFRAME(BlendingKeyframe, isBlendingKeyframe());
