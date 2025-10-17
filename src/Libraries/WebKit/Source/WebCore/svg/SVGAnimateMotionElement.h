/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 24, 2025.
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

#include "Path.h"
#include "SVGAnimationElement.h"

namespace WebCore {

class AffineTransform;
            
class SVGAnimateMotionElement final : public SVGAnimationElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGAnimateMotionElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGAnimateMotionElement);
public:
    static Ref<SVGAnimateMotionElement> create(const QualifiedName&, Document&);
    void updateAnimationPath();

private:
    SVGAnimateMotionElement(const QualifiedName&, Document&);

    bool hasValidAttributeType() const override;
    bool hasValidAttributeName() const override;

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;

    void startAnimation() override;
    void stopAnimation(SVGElement* targetElement) override;
    bool setFromAndToValues(const String& fromString, const String& toString) override;
    bool setFromAndByValues(const String& fromString, const String& byString) override;
    bool setToAtEndOfDurationValue(const String& toAtEndOfDurationString) override;
    void calculateAnimatedValue(float percentage, unsigned repeatCount) override;
    void applyResultsToTarget() override;
    std::optional<float> calculateDistance(const String& fromString, const String& toString) override;

    enum RotateMode {
        RotateAngle,
        RotateAuto,
        RotateAutoReverse
    };
    RotateMode rotateMode() const;
    void buildTransformForProgress(AffineTransform*, float percentage);

    void updateAnimationMode() override;
    void childrenChanged(const ChildChange&) final;

    // Note: we do not support percentage values for to/from coords as the spec implies we should (opera doesn't either)
    FloatPoint m_fromPoint;
    FloatPoint m_toPoint;
    std::optional<FloatPoint> m_toPointAtEndOfDuration;

    Path m_path;
    Path m_animationPath;
};
    
} // namespace WebCore
