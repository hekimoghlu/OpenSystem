/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 25, 2025.
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
#include "config.h"
#include "MeterPart.h"

#include "ControlFactory.h"

namespace WebCore {

Ref<MeterPart> MeterPart::create()
{
    return adoptRef(*new MeterPart(GaugeRegion::EvenLessGood, 0, 0, 0));
}

Ref<MeterPart> MeterPart::create(GaugeRegion gaugeRegion, double value, double minimum, double maximum)
{
    return adoptRef(*new MeterPart(gaugeRegion, value, minimum, maximum));
}

MeterPart::MeterPart(GaugeRegion gaugeRegion, double value, double minimum, double maximum)
    : ControlPart(StyleAppearance::Meter)
    , m_gaugeRegion(gaugeRegion)
    , m_value(value)
    , m_minimum(minimum)
    , m_maximum(maximum)
{
}

std::unique_ptr<PlatformControl> MeterPart::createPlatformControl()
{
    return controlFactory().createPlatformMeter(*this);
}

} // namespace WebCore
