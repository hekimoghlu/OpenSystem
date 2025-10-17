/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 24, 2023.
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

#include "ControlPart.h"

namespace WebCore {

class MeterPart : public ControlPart {
public:
    enum class GaugeRegion : uint8_t {
        Optimum,
        Suboptimal,
        EvenLessGood
    };

    static Ref<MeterPart> create();
    WEBCORE_EXPORT static Ref<MeterPart> create(GaugeRegion, double value, double minimum, double maximum);

    GaugeRegion gaugeRegion() const { return m_gaugeRegion; }
    void setGaugeRegion(GaugeRegion gaugeRegion) { m_gaugeRegion = gaugeRegion; }

    double value() const { return m_value; }
    void setValue(double value) { m_value = value; }

    double minimum() const { return m_minimum; }
    void setMinimum(double minimum) { m_minimum = minimum; }

    double maximum() const { return m_maximum; }
    void setMaximum(double maximum) { m_maximum = maximum; }

private:
    MeterPart(GaugeRegion, double value, double minimum, double maximum);

    std::unique_ptr<PlatformControl> createPlatformControl() override;

    GaugeRegion m_gaugeRegion;
    double m_value;
    double m_minimum;
    double m_maximum;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CONTROL_PART(Meter)
