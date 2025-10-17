/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 13, 2022.
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
#ifndef ZeroPole_h
#define ZeroPole_h

#include <wtf/TZoneMalloc.h>

namespace WebCore {

// ZeroPole is a simple filter with one zero and one pole.

class ZeroPole final {
    WTF_MAKE_TZONE_ALLOCATED(ZeroPole);
public:
    ZeroPole()
        : m_zero(0)
        , m_pole(0)
        , m_lastX(0)
        , m_lastY(0)
    {
    }

    void process(std::span<const float> source, std::span<float> destination, unsigned framesToProcess);

    // Reset filter state.
    void reset() { m_lastX = 0; m_lastY = 0; }
    
    void setZero(float zero) { m_zero = zero; }
    void setPole(float pole) { m_pole = pole; }
    
    float zero() const { return m_zero; }
    float pole() const { return m_pole; }

private:
    float m_zero;
    float m_pole;
    float m_lastX;
    float m_lastY;
};

} // namespace WebCore

#endif // ZeroPole_h
