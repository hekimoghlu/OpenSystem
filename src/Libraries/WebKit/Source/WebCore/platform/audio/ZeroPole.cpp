/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 29, 2021.
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

#if ENABLE(WEB_AUDIO)

#include "ZeroPole.h"

#include "DenormalDisabler.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ZeroPole);

void ZeroPole::process(std::span<const float> source, std::span<float> destination, unsigned framesToProcess)
{
    float zero = m_zero;
    float pole = m_pole;

    // Gain compensation to make 0dB @ 0Hz
    const float k1 = 1 / (1 - zero);
    const float k2 = 1 - pole;
    
    // Member variables to locals.
    float lastX = m_lastX;
    float lastY = m_lastY;

    for (unsigned i = 0; i < framesToProcess; ++i) {
        float input = source[i];

        // Zero
        float output1 = k1 * (input - zero * lastX);
        lastX = input;

        // Pole
        float output2 = k2 * output1 + pole * lastY;
        lastY = output2;

        destination[i] = output2;
    }
    
    // Locals to member variables. Flush denormals here so we don't
    // slow down the inner loop above.
    m_lastX = DenormalDisabler::flushDenormalFloatToZero(lastX);
    m_lastY = DenormalDisabler::flushDenormalFloatToZero(lastY);
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
