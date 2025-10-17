/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 22, 2024.
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

#include "FloatPoint.h"
#include "SVGPathByteStream.h"
#include "SVGPathSource.h"
#include <wtf/StdLibExtras.h>
#include <wtf/text/ParsingUtilities.h>

namespace WebCore {

class SVGPathByteStreamSource final : public SVGPathSource {
public:
    explicit SVGPathByteStreamSource(const SVGPathByteStream&);

private:
    bool hasMoreData() const final;
    bool moveToNextToken() final { return true; }
    SVGPathSegType nextCommand(SVGPathSegType) final;

    std::optional<SVGPathSegType> parseSVGSegmentType() final;
    std::optional<MoveToSegment> parseMoveToSegment(FloatPoint) final;
    std::optional<LineToSegment> parseLineToSegment(FloatPoint) final;
    std::optional<LineToHorizontalSegment> parseLineToHorizontalSegment(FloatPoint) final;
    std::optional<LineToVerticalSegment> parseLineToVerticalSegment(FloatPoint) final;
    std::optional<CurveToCubicSegment> parseCurveToCubicSegment(FloatPoint) final;
    std::optional<CurveToCubicSmoothSegment> parseCurveToCubicSmoothSegment(FloatPoint) final;
    std::optional<CurveToQuadraticSegment> parseCurveToQuadraticSegment(FloatPoint) final;
    std::optional<CurveToQuadraticSmoothSegment> parseCurveToQuadraticSmoothSegment(FloatPoint) final;
    std::optional<ArcToSegment> parseArcToSegment(FloatPoint) final;

#if COMPILER(MSVC)
#pragma warning(disable: 4701)
#endif
    template<typename DataType>
    DataType readType()
    {
        DataType data;
        size_t dataSize = sizeof(DataType);
        memcpySpan(asMutableByteSpan(data), consumeSpan(m_streamCurrent, dataSize));
        return data;
    }

    bool readFlag()
    {
        return readType<bool>();
    }

    float readFloat()
    {
        return readType<float>();
    }

    SVGPathSegType readSVGSegmentType()
    {
        static_assert(std::is_same_v<std::underlying_type_t<SVGPathSegType>, uint8_t>);
        return static_cast<SVGPathSegType>(readType<uint8_t>());
    }

    FloatPoint readFloatPoint()
    {
        return readType<FloatPoint>();
    }

    std::span<const uint8_t> m_streamCurrent;
};

} // namespace WebCore
