/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 15, 2023.
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

#include "AudioArray.h"
#include "ExceptionOr.h"
#include "FloatPoint3D.h"
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class AudioParam;
class BaseAudioContext;

// AudioListener maintains the state of the listener in the audio scene as defined in the OpenAL specification.

class AudioListener : public RefCounted<AudioListener> {
public:
    static Ref<AudioListener> create(BaseAudioContext& context)
    {
        return adoptRef(*new AudioListener(context));
    }
    ~AudioListener();

    AudioParam& positionX() { return m_positionX.get(); }
    AudioParam& positionY() { return m_positionY.get(); }
    AudioParam& positionZ() { return m_positionZ.get(); }
    AudioParam& forwardX() { return m_forwardX.get(); }
    AudioParam& forwardY() { return m_forwardY.get(); }
    AudioParam& forwardZ() { return m_forwardZ.get(); }
    AudioParam& upX() { return m_upX.get(); }
    AudioParam& upY() { return m_upY.get(); }
    AudioParam& upZ() { return m_upZ.get(); }

    // Position
    ExceptionOr<void> setPosition(float x, float y, float z);
    FloatPoint3D position() const;

    // Orientation
    ExceptionOr<void> setOrientation(float x, float y, float z, float upX, float upY, float upZ);
    FloatPoint3D orientation() const;

    FloatPoint3D upVector() const;

    bool hasSampleAccurateValues() const;
    bool shouldUseARate() const;

    std::span<const float> positionXValues(size_t framesToProcess);
    std::span<const float> positionYValues(size_t framesToProcess);
    std::span<const float> positionZValues(size_t framesToProcess);

    std::span<const float> forwardXValues(size_t framesToProcess);
    std::span<const float> forwardYValues(size_t framesToProcess);
    std::span<const float> forwardZValues(size_t framesToProcess);

    std::span<const float> upXValues(size_t framesToProcess);
    std::span<const float> upYValues(size_t framesToProcess);
    std::span<const float> upZValues(size_t framesToProcess);

    void updateValuesIfNeeded(size_t framesToProcess);

    void updateDirtyState();
    bool isPositionDirty() const { return m_isPositionDirty; }
    bool isOrientationDirty() const { return m_isOrientationDirty; }
    bool isUpVectorDirty() const { return m_isUpVectorDirty; }

protected:
    explicit AudioListener(BaseAudioContext&);

private:

    Ref<AudioParam> m_positionX;
    Ref<AudioParam> m_positionY;
    Ref<AudioParam> m_positionZ;
    Ref<AudioParam> m_forwardX;
    Ref<AudioParam> m_forwardY;
    Ref<AudioParam> m_forwardZ;
    Ref<AudioParam> m_upX;
    Ref<AudioParam> m_upY;
    Ref<AudioParam> m_upZ;

    // Last time that the automations were updated.
    double m_lastUpdateTime { -1 };

    AudioFloatArray m_positionXValues;
    AudioFloatArray m_positionYValues;
    AudioFloatArray m_positionZValues;

    AudioFloatArray m_forwardXValues;
    AudioFloatArray m_forwardYValues;
    AudioFloatArray m_forwardZValues;

    AudioFloatArray m_upXValues;
    AudioFloatArray m_upYValues;
    AudioFloatArray m_upZValues;

    FloatPoint3D m_lastPosition;
    FloatPoint3D m_lastOrientation;
    FloatPoint3D m_lastUpVector;
    bool m_isPositionDirty { false };
    bool m_isOrientationDirty { false };
    bool m_isUpVectorDirty { false };
};

} // namespace WebCore
