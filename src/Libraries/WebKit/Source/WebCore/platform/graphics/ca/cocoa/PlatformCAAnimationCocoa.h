/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 3, 2023.
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
#ifndef PlatformCAAnimationCocoa_h
#define PlatformCAAnimationCocoa_h

#include "PlatformCAAnimation.h"

#include <wtf/RetainPtr.h>

OBJC_CLASS CAMediaTimingFunction;
OBJC_CLASS CAAnimation;
OBJC_CLASS CAPropertyAnimation;
OBJC_CLASS NSString;

typedef CAAnimation* PlatformAnimationRef;

namespace WebCore {

WEBCORE_EXPORT NSString* toCAFillModeType(PlatformCAAnimation::FillModeType);
WEBCORE_EXPORT NSString* toCAValueFunctionType(PlatformCAAnimation::ValueFunctionType);
WEBCORE_EXPORT CAMediaTimingFunction* toCAMediaTimingFunction(const TimingFunction&, bool reverse);

bool hasExplicitBeginTime(CAAnimation *);
void setHasExplicitBeginTime(CAAnimation *, bool);

class PlatformCAAnimationCocoa final : public PlatformCAAnimation {
public:
    static Ref<PlatformCAAnimation> create(AnimationType, const String& keyPath);
    WEBCORE_EXPORT static Ref<PlatformCAAnimation> create(PlatformAnimationRef);

    virtual ~PlatformCAAnimationCocoa();

    bool isPlatformCAAnimationCocoa() const override { return true; }

    Ref<PlatformCAAnimation> copy() const override;

    PlatformAnimationRef platformAnimation() const;
    
    String keyPath() const override;
    
    CFTimeInterval beginTime() const override;
    void setBeginTime(CFTimeInterval) override;
    
    CFTimeInterval duration() const override;
    void setDuration(CFTimeInterval) override;
    
    float speed() const override;
    void setSpeed(float) override;

    CFTimeInterval timeOffset() const override;
    void setTimeOffset(CFTimeInterval) override;

    float repeatCount() const override;
    void setRepeatCount(float) override;

    bool autoreverses() const override;
    void setAutoreverses(bool) override;

    FillModeType fillMode() const override;
    void setFillMode(FillModeType) override;
    
    void setTimingFunction(const TimingFunction*, bool reverse = false) override;
    void copyTimingFunctionFrom(const PlatformCAAnimation&) override;

    bool isRemovedOnCompletion() const override;
    void setRemovedOnCompletion(bool) override;

    bool isAdditive() const override;
    void setAdditive(bool) override;

    ValueFunctionType valueFunction() const override;
    void setValueFunction(ValueFunctionType) override;

    // Basic-animation properties.
    void setFromValue(float) override;
    void setFromValue(const WebCore::TransformationMatrix&) override;
    void setFromValue(const FloatPoint3D&) override;
    void setFromValue(const WebCore::Color&) override;
    void setFromValue(const FilterOperation&) override;
    void copyFromValueFrom(const PlatformCAAnimation&) override;

    void setToValue(float) override;
    void setToValue(const WebCore::TransformationMatrix&) override;
    void setToValue(const FloatPoint3D&) override;
    void setToValue(const WebCore::Color&) override;
    void setToValue(const FilterOperation&) override;
    void copyToValueFrom(const PlatformCAAnimation&) override;

    // Keyframe-animation properties.
    void setValues(const Vector<float>&) override;
    void setValues(const Vector<WebCore::TransformationMatrix>&) override;
    void setValues(const Vector<FloatPoint3D>&) override;
    void setValues(const Vector<WebCore::Color>&) override;
    void setValues(const Vector<Ref<FilterOperation>>&) override;
    void copyValuesFrom(const PlatformCAAnimation&) override;

    void setKeyTimes(const Vector<float>&) override;
    void copyKeyTimesFrom(const PlatformCAAnimation&) override;

    void setTimingFunctions(const Vector<Ref<const TimingFunction>>&, bool reverse) override;
    void copyTimingFunctionsFrom(const PlatformCAAnimation&) override;

    // Animation group properties.
    void setAnimations(const Vector<RefPtr<PlatformCAAnimation>>&) final;
    void copyAnimationsFrom(const PlatformCAAnimation&) final;

private:
    PlatformCAAnimationCocoa(AnimationType, const String& keyPath);
    PlatformCAAnimationCocoa(PlatformAnimationRef);

    RetainPtr<CAAnimation> m_animation;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CAANIMATION(WebCore::PlatformCAAnimationCocoa, isPlatformCAAnimationCocoa())

#endif // PlatformCAAnimationCocoa_h
