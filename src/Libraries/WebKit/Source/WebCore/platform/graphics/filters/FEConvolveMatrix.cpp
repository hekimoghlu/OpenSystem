/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 29, 2022.
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
#include "FEConvolveMatrix.h"

#include "FEConvolveMatrixSoftwareApplier.h"
#include "Filter.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

Ref<FEConvolveMatrix> FEConvolveMatrix::create(const IntSize& kernelSize, float divisor, float bias, const IntPoint& targetOffset, EdgeModeType edgeMode, const FloatPoint& kernelUnitLength, bool preserveAlpha, const Vector<float>& kernelMatrix, DestinationColorSpace colorSpace)
{
    return adoptRef(*new FEConvolveMatrix(kernelSize, divisor, bias, targetOffset, edgeMode, kernelUnitLength, preserveAlpha, kernelMatrix, colorSpace));
}

FEConvolveMatrix::FEConvolveMatrix(const IntSize& kernelSize, float divisor, float bias, const IntPoint& targetOffset, EdgeModeType edgeMode, const FloatPoint& kernelUnitLength, bool preserveAlpha, const Vector<float>& kernelMatrix, DestinationColorSpace colorSpace)
    : FilterEffect(FilterEffect::Type::FEConvolveMatrix, colorSpace)
    , m_kernelSize(kernelSize)
    , m_divisor(divisor)
    , m_bias(bias)
    , m_targetOffset(targetOffset)
    , m_edgeMode(edgeMode)
    , m_kernelUnitLength(kernelUnitLength)
    , m_preserveAlpha(preserveAlpha)
    , m_kernelMatrix(kernelMatrix)
{
    ASSERT(m_kernelSize.width() > 0);
    ASSERT(m_kernelSize.height() > 0);
    ASSERT(IntRect(IntPoint::zero(), kernelSize).contains(targetOffset));
}

bool FEConvolveMatrix::operator==(const FEConvolveMatrix& other) const
{
    return FilterEffect::operator==(other)
        && m_kernelSize == other.m_kernelSize
        && m_divisor == other.m_divisor
        && m_bias == other.m_bias
        && m_targetOffset == other.m_targetOffset
        && m_edgeMode == other.m_edgeMode
        && m_kernelUnitLength == other.m_kernelUnitLength
        && m_preserveAlpha == other.m_preserveAlpha
        && m_kernelMatrix == other.m_kernelMatrix;
}

void FEConvolveMatrix::setKernelSize(const IntSize& kernelSize)
{
    ASSERT(kernelSize.width() > 0);
    ASSERT(kernelSize.height() > 0);
    m_kernelSize = kernelSize;
}

void FEConvolveMatrix::setKernel(const Vector<float>& kernel)
{
    m_kernelMatrix = kernel; 
}

bool FEConvolveMatrix::setDivisor(float divisor)
{
    ASSERT(divisor);
    if (m_divisor == divisor)
        return false;
    m_divisor = divisor;
    return true;
}

bool FEConvolveMatrix::setBias(float bias)
{
    if (m_bias == bias)
        return false;
    m_bias = bias;
    return true;
}

bool FEConvolveMatrix::setTargetOffset(const IntPoint& targetOffset)
{
    if (m_targetOffset == targetOffset)
        return false;
    m_targetOffset = targetOffset;
    return true;
}

bool FEConvolveMatrix::setEdgeMode(EdgeModeType edgeMode)
{
    if (m_edgeMode == edgeMode)
        return false;
    m_edgeMode = edgeMode;
    return true;
}

bool FEConvolveMatrix::setKernelUnitLength(const FloatPoint& kernelUnitLength)
{
    ASSERT(kernelUnitLength.x() > 0);
    ASSERT(kernelUnitLength.y() > 0);
    if (m_kernelUnitLength == kernelUnitLength)
        return false;
    m_kernelUnitLength = kernelUnitLength;
    return true;
}

bool FEConvolveMatrix::setPreserveAlpha(bool preserveAlpha)
{
    if (m_preserveAlpha == preserveAlpha)
        return false;
    m_preserveAlpha = preserveAlpha;
    return true;
}

FloatRect FEConvolveMatrix::calculateImageRect(const Filter& filter, std::span<const FloatRect>, const FloatRect& primitiveSubregion) const
{
    return filter.maxEffectRect(primitiveSubregion);
}

std::unique_ptr<FilterEffectApplier> FEConvolveMatrix::createSoftwareApplier() const
{
    return FilterEffectApplier::create<FEConvolveMatrixSoftwareApplier>(*this);
}

static TextStream& operator<<(TextStream& ts, const EdgeModeType& type)
{
    switch (type) {
    case EdgeModeType::Unknown:
        ts << "UNKNOWN";
        break;
    case EdgeModeType::Duplicate:
        ts << "DUPLICATE";
        break;
    case EdgeModeType::Wrap:
        ts << "WRAP";
        break;
    case EdgeModeType::None:
        ts << "NONE";
        break;
    }
    return ts;
}

TextStream& FEConvolveMatrix::externalRepresentation(TextStream& ts, FilterRepresentation representation) const
{
    ts << indent << "[feConvolveMatrix";
    FilterEffect::externalRepresentation(ts, representation);

    ts << " order=\"" << m_kernelSize << "\"";
    ts << " kernelMatrix=\"" << m_kernelMatrix  << "\"";
    ts << " divisor=\"" << m_divisor << "\"";
    ts << " bias=\"" << m_bias << "\"";
    ts << " target=\"" << m_targetOffset << "\"";
    ts << " edgeMode=\"" << m_edgeMode << "\"";
    ts << " kernelUnitLength=\"" << m_kernelUnitLength << "\"";
    ts << " preserveAlpha=\"" << m_preserveAlpha << "\"";

    ts << "]\n";
    return ts;
}

}; // namespace WebCore
