/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 27, 2024.
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

#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class IIRFilter {
    WTF_MAKE_TZONE_ALLOCATED(IIRFilter);
public:
    static constexpr size_t maxOrder { 20 };
    IIRFilter(const Vector<double>& feedforward, const Vector<double>& feedback);

    void reset();

    void process(std::span<const float> source, std::span<float> destination);
    void getFrequencyResponse(unsigned length, std::span<const float> frequency, std::span<float> magResponse, std::span<float> phaseResponse);
    double tailTime(double sampleRate, bool isFilterStable);

    const Vector<double>& feedforward() const { return m_feedforward; }
    const Vector<double>& feedback() const { return m_feedback; }

private:
    // Filter memory
    //
    // For simplicity, we assume |m_xBuffer| and |m_yBuffer| have the same length,
    // and the length is a power of two. Since the number of coefficients has a
    // fixed upper length, the size of xBuffer and yBuffer is fixed. |m_xBuffer|
    // holds the old input values and |m_yBuffer| holds the old output values
    // needed to compute the new output value.
    //
    // m_yBuffer[m_bufferIndex] holds the most recent output value, say, y[n].
    // Then m_yBuffer[m_bufferIndex - k] is y[n - k]. Similarly for m_xBuffer.
    //
    // To minimize roundoff, these arrays are double's instead of floats.
    Vector<double> m_xBuffer;
    Vector<double> m_yBuffer;

    // Index into the xBuffer and yBuffer arrays where the most current x and y
    // values should be stored. xBuffer[bufferIndex] corresponds to x[n], the
    // current x input value and yBuffer[bufferIndex] is where y[n], the current
    // output value.
    size_t m_bufferIndex { 0 };

    // Those Vectors are owned by the IIRProcess, which owns the IIRFilters via
    // the IIRDSPKernels. This is a memory optimization to avoid having copies
    // of these vectors in each IIRDSPKernel / IIRFilter.
    const Vector<double>& m_feedforward;
    const Vector<double>& m_feedback;
};

} // namespace WebCore
