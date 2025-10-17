/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 12, 2023.
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

#if USE(GSTREAMER)

#include "GStreamerCommon.h"
#include "GStreamerQuirks.h"
#include "MediaPlayerPrivateGStreamer.h"
#include <wtf/Vector.h>

namespace WebCore {

class GStreamerQuirkBroadcomBase : public GStreamerQuirk {
public:
    GStreamerQuirkBroadcomBase();

    bool needsBufferingPercentageCorrection() const { return true; }
    ASCIILiteral queryBufferingPercentage(MediaPlayerPrivateGStreamer*, const GRefPtr<GstQuery>&) const;
    int correctBufferingPercentage(MediaPlayerPrivateGStreamer*, int originalBufferingPercentage, GstBufferingMode) const;
    void resetBufferingPercentage(MediaPlayerPrivateGStreamer*, int bufferingPercentage) const;
    void setupBufferingPercentageCorrection(MediaPlayerPrivateGStreamer*, GstState currentState, GstState newState, GRefPtr<GstElement>&&) const;

protected:
    class MovingAverage {
    public:
        MovingAverage(size_t length)
            : m_values(length)
        {
            // Ensure that the sum in accumulate() can't ever overflow, considering that the highest value
            // for stored percentages is 100.
            ASSERT(length < INT_MAX / 100);
        }

        void reset(int value)
        {
            ASSERT(value <= 100);
            for (size_t i = 0; i < m_values.size(); i++)
                m_values[i] = value;
        }

        int accumulate(int value)
        {
            ASSERT(value <= 100);
            int sum = 0;
            for (size_t i = 1; i < m_values.size(); i++) {
                m_values[i - 1] = m_values[i];
                sum += m_values[i - 1];
            }
            m_values[m_values.size() - 1] = value;
            sum += value;
            return sum / m_values.size();
        }
    private:
        Vector<int> m_values;
    };

    using GStreamerQuirkBase::GStreamerQuirkState;

    class GStreamerQuirkBroadcomBaseState : public GStreamerQuirkState {
    public:
        GStreamerQuirkBroadcomBaseState() = default;
        virtual ~GStreamerQuirkBroadcomBaseState() = default;

        GRefPtr<GstElement> m_audfilter;
        GRefPtr<GstElement> m_vidfilter;
        GRefPtr<GstElement> m_multiqueue;
        GRefPtr<GstElement> m_queue2;
        MovingAverage m_streamBufferingLevelMovingAverage { 10 };
    };

    virtual GStreamerQuirkBroadcomBaseState& ensureState(MediaPlayerPrivateGStreamer*) const;
};

} // namespace WebCore

#endif // USE(GSTREAMER)
