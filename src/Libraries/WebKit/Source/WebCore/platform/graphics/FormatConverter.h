/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 26, 2022.
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

#if ENABLE(WEBGL)

#include "GraphicsContextGL.h"
#include "IntRect.h"
#include <wtf/MallocSpan.h>
#include <wtf/StdLibExtras.h>

namespace WebCore {

class FormatConverter {
public:
    FormatConverter(
        const IntRect& sourceDataSubRectangle,
        int depth,
        int unpackImageHeight,
        std::span<const uint8_t> source,
        std::span<uint8_t> destinationCursor,
        std::span<uint8_t> destination,
        int srcStride,
        int srcRowOffset,
        int dstStride)
            : m_srcSubRectangle(sourceDataSubRectangle)
            , m_depth(depth)
            , m_unpackImageHeight(unpackImageHeight)
            , m_source(source)
            , m_destinationCursor(destinationCursor)
            , m_destination(destination)
            , m_srcStride(srcStride)
            , m_srcRowOffset(srcRowOffset)
            , m_dstStride(dstStride)
            , m_success(false)
    {
        const unsigned MaxNumberOfComponents = 4;
        const unsigned MaxBytesPerComponent  = 4;
        m_unpackedIntermediateSrcData = MallocSpan<uint8_t>::malloc(Checked<size_t>(m_srcSubRectangle.width()) * MaxNumberOfComponents * MaxBytesPerComponent);
        ASSERT(m_unpackedIntermediateSrcData);
    }

    void convert(GraphicsContextGL::DataFormat srcFormat, GraphicsContextGL::DataFormat dstFormat, GraphicsContextGL::AlphaOp);
    bool success() const { return m_success; }

private:
    template<GraphicsContextGL::DataFormat SrcFormat>
    ALWAYS_INLINE void convert(GraphicsContextGL::DataFormat dstFormat, GraphicsContextGL::AlphaOp);

    template<GraphicsContextGL::DataFormat SrcFormat, GraphicsContextGL::DataFormat DstFormat>
    ALWAYS_INLINE void convert(GraphicsContextGL::AlphaOp);

    template<GraphicsContextGL::DataFormat SrcFormat, GraphicsContextGL::DataFormat DstFormat, GraphicsContextGL::AlphaOp alphaOp>
    ALWAYS_INLINE void convert();

    const IntRect& m_srcSubRectangle;
    const int m_depth;
    const int m_unpackImageHeight;
    std::span<const uint8_t> m_source;
    std::span<uint8_t> m_destinationCursor;
    std::span<uint8_t> m_destination;
    const int m_srcStride, m_srcRowOffset, m_dstStride;
    bool m_success;
    MallocSpan<uint8_t> m_unpackedIntermediateSrcData;
};

} // namespace WebCore

#endif // ENABLE(WEBGL)
