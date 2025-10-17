/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 17, 2025.
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

#include "AudioContext.h"
#include "AudioParamTimeline.h"
#include "AudioSummingJunction.h"
#include "AutomationRate.h"
#include <JavaScriptCore/Forward.h>
#include <sys/types.h>
#include <wtf/LoggerHelper.h>
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class AudioNodeOutput;

enum class AutomationRateMode : bool { Fixed, Variable };

class AudioParam final
    : public AudioSummingJunction
    , public RefCounted<AudioParam>
#if !RELEASE_LOG_DISABLED
    , private LoggerHelper
#endif
{
public:
    static constexpr double SmoothingConstant = 0.05;
    static constexpr double SnapThreshold = 0.001;

    static Ref<AudioParam> create(BaseAudioContext& context, const String& name, float defaultValue, float minValue, float maxValue, AutomationRate automationRate, AutomationRateMode automationRateMode = AutomationRateMode::Variable)
    {
        return adoptRef(*new AudioParam(context, name, defaultValue, minValue, maxValue, automationRate, automationRateMode));
    }

    // AudioSummingJunction
    bool canUpdateState() override { return true; }
    void didUpdate() override { }

    // Intrinsic value.
    float value();
    void setValue(float);

    float valueForBindings() const;
    ExceptionOr<void> setValueForBindings(float);

    AutomationRate automationRate() const { return m_automationRate; }
    ExceptionOr<void> setAutomationRate(AutomationRate);

    // Final value for k-rate parameters, otherwise use calculateSampleAccurateValues() for a-rate.
    // Must be called in the audio thread.
    float finalValue();

    const String& name() const { return m_name; }

    float minValue() const { return m_minValue; }
    float maxValue() const { return m_maxValue; }
    float defaultValue() const { return m_defaultValue; }

    // Value smoothing:

    // When a new value is set with setValue(), in our internal use of the parameter we don't immediately jump to it.
    // Instead we smoothly approach this value to avoid glitching.
    float smoothedValue();

    // Smoothly exponentially approaches to (de-zippers) the desired value.
    // Returns true if smoothed value has already snapped exactly to value.
    bool smooth();

    void resetSmoothedValue() { m_smoothedValue = m_value; }

    // Parameter automation.    
    ExceptionOr<AudioParam&> setValueAtTime(float value, double startTime);
    ExceptionOr<AudioParam&> linearRampToValueAtTime(float value, double endTime);
    ExceptionOr<AudioParam&> exponentialRampToValueAtTime(float value, double endTime);
    ExceptionOr<AudioParam&> setTargetAtTime(float target, double startTime, float timeConstant);
    ExceptionOr<AudioParam&> setValueCurveAtTime(Vector<float>&& curve, double startTime, double duration);
    ExceptionOr<AudioParam&> cancelScheduledValues(double cancelTime);
    ExceptionOr<AudioParam&> cancelAndHoldAtTime(double cancelTime);

    bool hasSampleAccurateValues() const;
    
    // Calculates numberOfValues parameter values starting at the context's current time.
    // Must be called in the context's render thread.
    void calculateSampleAccurateValues(std::span<float> values);

    // Connect an audio-rate signal to control this parameter.
    void connect(AudioNodeOutput*);
    void disconnect(AudioNodeOutput*);

protected:
    AudioParam(BaseAudioContext&, const String&, float defaultValue, float minValue, float maxValue, AutomationRate, AutomationRateMode);

private:
    // sampleAccurate corresponds to a-rate (audio rate) vs. k-rate in the Web Audio specification.
    void calculateFinalValues(std::span<float> values, bool sampleAccurate);
    void calculateTimelineValues(std::span<float> values);

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final { return m_logger.get(); }
    uint64_t logIdentifier() const final { return m_logIdentifier; }
    ASCIILiteral logClassName() const final { return "AudioParam"_s; }
    WTFLogChannel& logChannel() const final;
#endif
    
    String m_name;
    float m_value;
    float m_defaultValue;
    float m_minValue;
    float m_maxValue;
    AutomationRate m_automationRate;
    AutomationRateMode m_automationRateMode;

    // Smoothing (de-zippering)
    float m_smoothedValue;
    
    AudioParamTimeline m_timeline;
    Ref<AudioBus> m_summingBus;

#if !RELEASE_LOG_DISABLED
    mutable Ref<const Logger> m_logger;
    const uint64_t m_logIdentifier;
#endif
};

} // namespace WebCore
