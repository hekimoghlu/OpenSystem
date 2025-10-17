/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 25, 2023.
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

#if ENABLE(WEB_AUDIO)

#include "AudioContext.h"
#include "AudioListener.h"
#include "AudioNode.h"
#include "AudioParam.h"
#include "Cone.h"
#include "Distance.h"
#include "FloatPoint3D.h"
#include "Panner.h"
#include "PannerOptions.h"
#include <memory>
#include <wtf/HashSet.h>
#include <wtf/Lock.h>

namespace WebCore {

class HRTFDatabaseLoader;
class BaseAudioContext;

// PannerNode is an AudioNode with one input and one output.
// It positions a sound in 3D space, with the exact effect dependent on the panning model.
// It has a position and an orientation in 3D space which is relative to the position and orientation of the context's AudioListener.
// A distance effect will attenuate the gain as the position moves away from the listener.
// A cone effect will attenuate the gain as the orientation moves away from the listener.
// All of these effects follow the OpenAL specification very closely.

class PannerNode final : public AudioNode {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(PannerNode);
public:
    static ExceptionOr<Ref<PannerNode>> create(BaseAudioContext&, const PannerOptions& = { });

    virtual ~PannerNode();

    // AudioNode
    void process(size_t framesToProcess) override;
    void processOnlyAudioParams(size_t framesToProcess) final;

    // Listener
    AudioListener& listener();

    // Panning model
    PanningModelType panningModelForBindings() const WTF_IGNORES_THREAD_SAFETY_ANALYSIS { ASSERT(isMainThread()); return m_panningModel; }
    void setPanningModelForBindings(PanningModelType);

    // Position
    ExceptionOr<void> setPosition(float x, float y, float z);
    AudioParam& positionX() WTF_IGNORES_THREAD_SAFETY_ANALYSIS { ASSERT(isMainThread()); return m_positionX.get(); }
    AudioParam& positionY() WTF_IGNORES_THREAD_SAFETY_ANALYSIS { ASSERT(isMainThread()); return m_positionY.get(); }
    AudioParam& positionZ() WTF_IGNORES_THREAD_SAFETY_ANALYSIS { ASSERT(isMainThread()); return m_positionZ.get(); }

    // Orientation
    ExceptionOr<void> setOrientation(float x, float y, float z);
    AudioParam& orientationX() WTF_IGNORES_THREAD_SAFETY_ANALYSIS { ASSERT(isMainThread()); return m_orientationX.get(); }
    AudioParam& orientationY() WTF_IGNORES_THREAD_SAFETY_ANALYSIS { ASSERT(isMainThread()); return m_orientationY.get(); }
    AudioParam& orientationZ() WTF_IGNORES_THREAD_SAFETY_ANALYSIS { ASSERT(isMainThread()); return m_orientationZ.get(); }

    // Distance parameters
    DistanceModelType distanceModelForBindings() const;
    void setDistanceModelForBindings(DistanceModelType);

    double refDistanceForBindings() const WTF_IGNORES_THREAD_SAFETY_ANALYSIS { ASSERT(isMainThread()); return m_distanceEffect.refDistance(); }
    ExceptionOr<void> setRefDistanceForBindings(double);

    double maxDistanceForBindings() const WTF_IGNORES_THREAD_SAFETY_ANALYSIS { ASSERT(isMainThread()); return m_distanceEffect.maxDistance(); }
    ExceptionOr<void> setMaxDistanceForBindings(double);

    double rolloffFactorForBindings() const WTF_IGNORES_THREAD_SAFETY_ANALYSIS { ASSERT(isMainThread()); return m_distanceEffect.rolloffFactor(); }
    ExceptionOr<void> setRolloffFactorForBindings(double);

    // Sound cones - angles in degrees
    double coneInnerAngleForBindings() const WTF_IGNORES_THREAD_SAFETY_ANALYSIS { ASSERT(isMainThread()); return m_coneEffect.innerAngle(); }
    void setConeInnerAngleForBindings(double);

    double coneOuterAngleForBindings() const WTF_IGNORES_THREAD_SAFETY_ANALYSIS { ASSERT(isMainThread()); return m_coneEffect.outerAngle(); }
    void setConeOuterAngleForBindings(double);

    double coneOuterGainForBindings() const WTF_IGNORES_THREAD_SAFETY_ANALYSIS { ASSERT(isMainThread()); return m_coneEffect.outerGain(); }
    ExceptionOr<void> setConeOuterGainForBindings(double);
    
    ExceptionOr<void> setChannelCount(unsigned) final;
    ExceptionOr<void> setChannelCountMode(ChannelCountMode) final;

    double tailTime() const final;
    double latencyTime() const final;

private:
    PannerNode(BaseAudioContext&, const PannerOptions&);

    struct AzimuthElevation {
        double azimuth { 0. };
        double elevation { 0. };
    };
    static AzimuthElevation calculateAzimuthElevation(const FloatPoint3D& position, const FloatPoint3D& listenerPosition, const FloatPoint3D& listenerForward, const FloatPoint3D& listenerUp);
    static float calculateDistanceConeGain(const FloatPoint3D& position, const FloatPoint3D& orientation, const FloatPoint3D& listenerPosition, const DistanceEffect&, const ConeEffect&);

    // Returns the combined distance and cone gain attenuation.
    float distanceConeGain() WTF_REQUIRES_LOCK(m_processLock);

    bool requiresTailProcessing() const final;

    void invalidateCachedPropertiesIfNecessary() WTF_REQUIRES_LOCK(m_processLock);

    const AzimuthElevation& azimuthElevation() WTF_REQUIRES_LOCK(m_processLock);
    void processSampleAccurateValues(AudioBus* destination, const AudioBus* source, size_t framesToProcess) WTF_REQUIRES_LOCK(m_processLock);
    bool hasSampleAccurateValues() const WTF_REQUIRES_LOCK(m_processLock);
    bool shouldUseARate() const WTF_REQUIRES_LOCK(m_processLock);

    FloatPoint3D position() const WTF_REQUIRES_LOCK(m_processLock);
    FloatPoint3D orientation() const WTF_REQUIRES_LOCK(m_processLock);

    Ref<HRTFDatabaseLoader> m_hrtfDatabaseLoader;
    PanningModelType m_panningModel WTF_GUARDED_BY_LOCK(m_processLock);
    std::unique_ptr<Panner> m_panner WTF_GUARDED_BY_LOCK(m_processLock);

    // Gain
    DistanceEffect m_distanceEffect WTF_GUARDED_BY_LOCK(m_processLock);
    ConeEffect m_coneEffect WTF_GUARDED_BY_LOCK(m_processLock);
    
    Ref<AudioParam> m_positionX WTF_GUARDED_BY_LOCK(m_processLock);
    Ref<AudioParam> m_positionY WTF_GUARDED_BY_LOCK(m_processLock);
    Ref<AudioParam> m_positionZ WTF_GUARDED_BY_LOCK(m_processLock);
    
    Ref<AudioParam> m_orientationX WTF_GUARDED_BY_LOCK(m_processLock);
    Ref<AudioParam> m_orientationY WTF_GUARDED_BY_LOCK(m_processLock);
    Ref<AudioParam> m_orientationZ WTF_GUARDED_BY_LOCK(m_processLock);

    mutable std::optional<AzimuthElevation> m_cachedAzimuthElevation WTF_GUARDED_BY_LOCK(m_processLock);
    mutable std::optional<float> m_cachedConeGain WTF_GUARDED_BY_LOCK(m_processLock);
    FloatPoint3D m_lastPosition WTF_GUARDED_BY_LOCK(m_processLock);
    FloatPoint3D m_lastOrientation WTF_GUARDED_BY_LOCK(m_processLock);

    // Synchronize process() with setting of the panning model, source's location
    // information, listener, distance parameters and sound cones.
    mutable Lock m_processLock;
};

} // namespace WebCore

#endif
