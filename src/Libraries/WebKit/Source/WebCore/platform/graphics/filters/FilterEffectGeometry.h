/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 1, 2025.
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

#include "FloatRect.h"
#include <wtf/ArgumentCoder.h>
#include <wtf/OptionSet.h>
#include <wtf/Ref.h>

namespace WebCore {

class FilterEffect;

class FilterEffectGeometry {
public:
    enum class Flags : uint8_t {
        HasX      = 1 << 0,
        HasY      = 1 << 1,
        HasWidth  = 1 << 2,
        HasHeight = 1 << 3
    };

    FilterEffectGeometry() = default;

    FilterEffectGeometry(const FloatRect& boundaries, OptionSet<Flags> flags)
        : m_boundaries(boundaries)
        , m_flags(flags)
    {
    }

    std::optional<float> x() const
    {
        if (m_flags.contains(Flags::HasX))
            return m_boundaries.x();
        return std::nullopt;
    }

    std::optional<float> y() const
    {
        if (m_flags.contains(Flags::HasY))
            return m_boundaries.y();
        return std::nullopt;
    }

    std::optional<float> width() const
    {
        if (m_flags.contains(Flags::HasWidth))
            return m_boundaries.width();
        return std::nullopt;
    }

    std::optional<float> height() const
    {
        if (m_flags.contains(Flags::HasHeight))
            return m_boundaries.height();
        return std::nullopt;
    }

private:
    friend struct IPC::ArgumentCoder<FilterEffectGeometry, void>;
    FloatRect m_boundaries;
    OptionSet<Flags> m_flags;
};

using FilterEffectGeometryMap = UncheckedKeyHashMap<Ref<FilterEffect>, FilterEffectGeometry>;

} // namespace WebCore
