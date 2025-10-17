/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 27, 2022.
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

#include "CSSNumericValue.h"
#include <wtf/Seconds.h>

namespace WebCore {

class WebAnimationTime {
public:
    WebAnimationTime() = default;
    WEBCORE_EXPORT WebAnimationTime(std::optional<Seconds>, std::optional<double>);

    WEBCORE_EXPORT WebAnimationTime(const Seconds&);
    WebAnimationTime(const CSSNumberish&);

    static WebAnimationTime fromMilliseconds(double);
    static WebAnimationTime fromPercentage(double);

    WEBCORE_EXPORT std::optional<Seconds> time() const;
    WEBCORE_EXPORT std::optional<double> percentage() const;

    bool isValid() const;
    bool isInfinity() const;
    bool isZero() const;
    bool isNaN() const;

    WebAnimationTime matchingZero() const;
    WebAnimationTime matchingEpsilon() const;
    WebAnimationTime matchingInfinity() const;

    bool approximatelyEqualTo(const WebAnimationTime&) const;
    bool approximatelyLessThan(const WebAnimationTime&) const;
    bool approximatelyGreaterThan(const WebAnimationTime&) const;

    WebAnimationTime operator+(const WebAnimationTime&) const;
    WebAnimationTime operator-(const WebAnimationTime&) const;
    double operator/(const WebAnimationTime&) const;
    WebAnimationTime& operator+=(const WebAnimationTime&);
    WebAnimationTime& operator-=(const WebAnimationTime&);
    bool operator<(const WebAnimationTime&) const;
    bool operator<=(const WebAnimationTime&) const;
    bool operator>(const WebAnimationTime&) const;
    bool operator>=(const WebAnimationTime&) const;
    bool operator==(const WebAnimationTime&) const;

    WebAnimationTime operator+(const Seconds&) const;
    WebAnimationTime operator-(const Seconds&) const;
    bool operator<(const Seconds&) const;
    bool operator<=(const Seconds&) const;
    bool operator>(const Seconds&) const;
    bool operator>=(const Seconds&) const;
    bool operator==(const Seconds&) const;

    WebAnimationTime operator*(double) const;
    WebAnimationTime operator/(double) const;

    WEBCORE_EXPORT operator Seconds() const;
    operator CSSNumberish() const;

    void dump(TextStream&) const;

private:
    enum class Type : uint8_t { Unknown, Time, Percentage };

    WebAnimationTime(Type, double);

    Type m_type { Type::Unknown };
    double m_value { 0 };
};

TextStream& operator<<(TextStream&, const WebAnimationTime&);

} // namespace WebCore
