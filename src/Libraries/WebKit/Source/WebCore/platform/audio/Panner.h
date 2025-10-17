/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 18, 2021.
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
#ifndef Panner_h
#define Panner_h

#include <memory>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class AudioBus;
class HRTFDatabaseLoader;

enum class PanningModelType {
    Equalpower,
    HRTF
};

// Abstract base class for panning a mono or stereo source.

class Panner {
    WTF_MAKE_TZONE_ALLOCATED(Panner);
public:
    static std::unique_ptr<Panner> create(PanningModelType, float sampleRate, HRTFDatabaseLoader*);

    virtual ~Panner() { };

    PanningModelType panningModel() const { return m_panningModel; }

    virtual void pan(double azimuth, double elevation, const AudioBus* inputBus, AudioBus* outputBus, size_t framesToProcess) = 0;
    virtual void panWithSampleAccurateValues(std::span<double> azimuth, std::span<double> elevation, const AudioBus* inputBus, AudioBus* outputBus, size_t framesToProcess) = 0;
    virtual void reset() = 0;

    virtual double tailTime() const = 0;
    virtual double latencyTime() const = 0;
    virtual bool requiresTailProcessing() const = 0;

protected:
    Panner(PanningModelType model)
        : m_panningModel(model)
    {
    }

    PanningModelType m_panningModel;
};

} // namespace WebCore

#endif // Panner_h
