/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 11, 2022.
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

#include <wtf/text/TextStream.h>

namespace WebCore {

struct BlendingContext;

class OffsetRotation {
public:
    constexpr OffsetRotation(bool hasAuto = false, float angle = 0) : m_angle(angle), m_hasAuto(hasAuto) { }

    bool hasAuto() const { return m_hasAuto; }
    float angle() const { return m_angle; }

    bool canBlend(const OffsetRotation&) const;
    OffsetRotation blend(const OffsetRotation&, const BlendingContext&) const;

    friend bool operator==(const OffsetRotation&, const OffsetRotation&) = default;

private:
    float m_angle { 0 };
    bool m_hasAuto { false };
};

WTF::TextStream& operator<<(WTF::TextStream&, const OffsetRotation&);

} // namespace WebCore
