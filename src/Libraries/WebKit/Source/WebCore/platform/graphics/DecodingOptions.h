/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 27, 2022.
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

#include "IntSize.h"
#include <optional>

namespace WebCore {

enum class DecodingMode : uint8_t {
    Auto,
    Synchronous,
    Asynchronous
};

class DecodingOptions {
public:
    DecodingOptions(DecodingMode decodingMode = DecodingMode::Synchronous, const std::optional<IntSize>& sizeForDrawing = std::nullopt)
        : m_decodingMode(decodingMode)
        , m_sizeForDrawing(sizeForDrawing)
    {
    }

    friend bool operator==(const DecodingOptions&, const DecodingOptions&) = default;

    DecodingMode decodingMode() const { return m_decodingMode; }
    bool isAuto() const { return m_decodingMode == DecodingMode::Auto; }
    bool isSynchronous() const { return m_decodingMode == DecodingMode::Synchronous; }
    bool isAsynchronous() const { return m_decodingMode == DecodingMode::Asynchronous; }

    std::optional<IntSize> sizeForDrawing() const { return m_sizeForDrawing; }
    bool hasFullSize() const { return !m_sizeForDrawing; }
    bool hasSizeForDrawing() const { return !!m_sizeForDrawing; }

    bool isCompatibleWith(const DecodingOptions& other) const
    {
        if (isAuto() || other.isAuto())
            return false;

        if (hasFullSize())
            return true;

        if (other.hasFullSize())
            return false;

        return sizeForDrawing()->maxDimension() >= other.sizeForDrawing()->maxDimension();
    }

private:
    DecodingMode m_decodingMode;
    std::optional<IntSize> m_sizeForDrawing;
};

TextStream& operator<<(TextStream&, DecodingMode);

} // namespace WebCore
