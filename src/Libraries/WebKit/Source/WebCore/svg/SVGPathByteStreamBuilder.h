/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 23, 2024.
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
#include "SVGPathConsumer.h"
#include <wtf/text/WTFString.h>

namespace WebCore {

enum class SVGPathSegType : uint8_t;

class SVGPathByteStreamBuilder final : public SVGPathConsumer {
public:
    SVGPathByteStreamBuilder(SVGPathByteStream&);

private:
    void incrementPathSegmentCount() final { }
    bool continueConsuming() final { return true; }

    // Used in UnalteredParsing/NormalizedParsing modes.
    void moveTo(const FloatPoint&, bool closed, PathCoordinateMode) final;
    void lineTo(const FloatPoint&, PathCoordinateMode) final;
    void curveToCubic(const FloatPoint&, const FloatPoint&, const FloatPoint&, PathCoordinateMode) final;
    void closePath() final;

    // Only used in UnalteredParsing mode.
    void lineToHorizontal(float, PathCoordinateMode) final;
    void lineToVertical(float, PathCoordinateMode) final;
    void curveToCubicSmooth(const FloatPoint&, const FloatPoint&, PathCoordinateMode) final;
    void curveToQuadratic(const FloatPoint&, const FloatPoint&, PathCoordinateMode) final;
    void curveToQuadraticSmooth(const FloatPoint&, PathCoordinateMode) final;
    void arcTo(float, float, float, bool largeArcFlag, bool sweepFlag, const FloatPoint&, PathCoordinateMode) final;

    template<typename DataType>
    void writeType(const DataType& data)
    {
        typedef union {
            DataType value;
            uint8_t bytes[sizeof(DataType)];
        } ByteType;

        ByteType type = { data };
        m_byteStream->append(std::span { type.bytes, sizeof(ByteType) });
    }

    void writeSegmentType(SVGPathSegType type)
    {
        static_assert(std::is_same_v<std::underlying_type_t<SVGPathSegType>, uint8_t>);
        m_byteStream->append(enumToUnderlyingType(type));
    }

    SingleThreadWeakRef<SVGPathByteStream> m_byteStream;
};

} // namespace WebCore
