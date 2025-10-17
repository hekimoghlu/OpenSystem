/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 1, 2022.
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
#ifndef Cone_h
#define Cone_h

#include "FloatPoint3D.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

// Cone gain is defined according to the OpenAL specification

class ConeEffect final {
    WTF_MAKE_TZONE_ALLOCATED(ConeEffect);
public:
    ConeEffect();

    // Returns scalar gain for the given source/listener positions/orientations
    double gain(FloatPoint3D sourcePosition, FloatPoint3D sourceOrientation, FloatPoint3D listenerPosition) const;

    // Angles in degrees
    void setInnerAngle(double innerAngle) { m_innerAngle = innerAngle; }
    double innerAngle() const { return m_innerAngle; }

    void setOuterAngle(double outerAngle) { m_outerAngle = outerAngle; }
    double outerAngle() const { return m_outerAngle; }

    void setOuterGain(double outerGain) { m_outerGain = outerGain; }
    double outerGain() const { return m_outerGain; }

protected:
    double m_innerAngle { 360 };
    double m_outerAngle { 360 };
    double m_outerGain { 0 };
};

} // namespace WebCore

#endif // Cone_h
