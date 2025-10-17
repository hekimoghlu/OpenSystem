/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 11, 2021.
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

#include "PlatformCAAnimationRemoteProperties.h"
#include <WebCore/PlatformCAAnimation.h>
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>

namespace WTF {
class TextStream;
};

OBJC_CLASS CALayer;

namespace WebKit {

class RemoteLayerTreeHost;

class PlatformCAAnimationRemote final : public WebCore::PlatformCAAnimation {
public:
    static Ref<PlatformCAAnimation> create(AnimationType, const String& keyPath);

    virtual ~PlatformCAAnimationRemote() { }

    bool isPlatformCAAnimationRemote() const override { return true; }

    Ref<PlatformCAAnimation> copy() const override;

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

    void setTimingFunction(const WebCore::TimingFunction*, bool reverse = false) override;
    void copyTimingFunctionFrom(const WebCore::PlatformCAAnimation&) override;

    bool isRemovedOnCompletion() const override;
    void setRemovedOnCompletion(bool) override;

    bool isAdditive() const override;
    void setAdditive(bool) override;

    ValueFunctionType valueFunction() const override;
    void setValueFunction(ValueFunctionType) override;

    // Basic-animation properties.
    void setFromValue(float) override;
    void setFromValue(const WebCore::TransformationMatrix&) override;
    void setFromValue(const WebCore::FloatPoint3D&) override;
    void setFromValue(const WebCore::Color&) override;
    void setFromValue(const WebCore::FilterOperation&) override;
    void copyFromValueFrom(const WebCore::PlatformCAAnimation&) override;

    void setToValue(float) override;
    void setToValue(const WebCore::TransformationMatrix&) override;
    void setToValue(const WebCore::FloatPoint3D&) override;
    void setToValue(const WebCore::Color&) override;
    void setToValue(const WebCore::FilterOperation&) override;
    void copyToValueFrom(const WebCore::PlatformCAAnimation&) override;

    // Keyframe-animation properties.
    void setValues(const Vector<float>&) override;
    void setValues(const Vector<WebCore::TransformationMatrix>&) override;
    void setValues(const Vector<WebCore::FloatPoint3D>&) override;
    void setValues(const Vector<WebCore::Color>&) override;
    void setValues(const Vector<Ref<WebCore::FilterOperation>>&) override;
    void copyValuesFrom(const WebCore::PlatformCAAnimation&) override;

    void setKeyTimes(const Vector<float>&) override;
    void copyKeyTimesFrom(const WebCore::PlatformCAAnimation&) override;

    void setTimingFunctions(const Vector<Ref<const WebCore::TimingFunction>>&, bool reverse) override;
    void copyTimingFunctionsFrom(const WebCore::PlatformCAAnimation&) override;

    // Animation group properties.
    void setAnimations(const Vector<RefPtr<PlatformCAAnimation>>&) final;
    void copyAnimationsFrom(const PlatformCAAnimation&) final;

    AnimationType animationType() const { return m_properties.animationType; }
    void setHasExplicitBeginTime(bool hasExplicitBeginTime) { m_properties.hasExplicitBeginTime = hasExplicitBeginTime; }
    bool hasExplicitBeginTime() const { return m_properties.hasExplicitBeginTime; }

    void didStart(CFTimeInterval beginTime) { m_properties.beginTime = beginTime; }

    using KeyframeValue = PlatformCAAnimationRemoteProperties::KeyframeValue;
    using Properties = PlatformCAAnimationRemoteProperties;

    const Properties& properties() const { return m_properties; }

    typedef Vector<std::pair<String, Properties>> AnimationsList;
    static void updateLayerAnimations(CALayer *, RemoteLayerTreeHost*, const AnimationsList& animationsToAdd, const HashSet<String>& animationsToRemove);

private:
    PlatformCAAnimationRemote(AnimationType, const String& keyPath);

    Properties m_properties;
};

WTF::TextStream& operator<<(WTF::TextStream&, const PlatformCAAnimationRemote::Properties&);

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_CAANIMATION(WebKit::PlatformCAAnimationRemote, isPlatformCAAnimationRemote())
