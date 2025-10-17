/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 4, 2022.
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
#include "FEGaussianBlur.h"

#include "FEGaussianBlurSoftwareApplier.h"
#include "Filter.h"
#include <wtf/text/TextStream.h>

#if USE(SKIA)
#include "FEGaussianBlurSkiaApplier.h"
#endif

namespace WebCore {

Ref<FEGaussianBlur> FEGaussianBlur::create(float x, float y, EdgeModeType edgeMode, DestinationColorSpace colorSpace)
{
    return adoptRef(*new FEGaussianBlur(x, y, edgeMode, colorSpace));
}

FEGaussianBlur::FEGaussianBlur(float x, float y, EdgeModeType edgeMode, DestinationColorSpace colorSpace)
    : FilterEffect(FilterEffect::Type::FEGaussianBlur, colorSpace)
    , m_stdX(x)
    , m_stdY(y)
    , m_edgeMode(edgeMode)
{
}

bool FEGaussianBlur::operator==(const FEGaussianBlur& other) const
{
    return FilterEffect::operator==(other)
        && m_stdX == other.m_stdX
        && m_stdY == other.m_stdY
        && m_edgeMode == other.m_edgeMode;
}

bool FEGaussianBlur::setStdDeviationX(float stdX)
{
    if (m_stdX == stdX)
        return false;
    m_stdX = stdX;
    return true;
}

bool FEGaussianBlur::setStdDeviationY(float stdY)
{
    if (m_stdY == stdY)
        return false;
    m_stdY = stdY;
    return true;
}

bool FEGaussianBlur::setEdgeMode(EdgeModeType edgeMode)
{
    if (m_edgeMode == edgeMode)
        return false;
    m_edgeMode = edgeMode;
    return true;
}

static inline float gaussianKernelFactor()
{
    return 3 / 4.f * sqrtf(2 * piFloat);
}

static int clampedToKernelSize(float value)
{
    static constexpr unsigned maxKernelSize = 500;

    // Limit the kernel size to 500. A bigger radius won't make a big difference for the result image but
    // inflates the absolute paint rect too much. This is compatible with Firefox' behavior.
    unsigned size = std::max<unsigned>(2, static_cast<unsigned>(floorf(value * gaussianKernelFactor() + 0.5f)));
    return clampTo<int>(std::min(size, maxKernelSize));
}
    
IntSize FEGaussianBlur::calculateUnscaledKernelSize(FloatSize stdDeviation)
{
    ASSERT(stdDeviation.width() >= 0 && stdDeviation.height() >= 0);
    IntSize kernelSize;

    if (stdDeviation.width())
        kernelSize.setWidth(clampedToKernelSize(stdDeviation.width()));

    if (stdDeviation.height())
        kernelSize.setHeight(clampedToKernelSize(stdDeviation.height()));

    return kernelSize;
}

IntSize FEGaussianBlur::calculateKernelSize(const Filter& filter, FloatSize stdDeviation)
{
    stdDeviation = filter.resolvedSize(stdDeviation);
    return calculateUnscaledKernelSize(filter.scaledByFilterScale(stdDeviation));
}

IntSize FEGaussianBlur::calculateOutsetSize(FloatSize stdDeviation)
{
    auto kernelSize = calculateUnscaledKernelSize(stdDeviation);

    // We take the half kernel size and multiply it with three, because we run box blur three times.
    return { 3 * kernelSize.width() / 2, 3 * kernelSize.height() / 2 };
}

FloatRect FEGaussianBlur::calculateImageRect(const Filter& filter, std::span<const FloatRect> inputImageRects, const FloatRect& primitiveSubregion) const
{
    auto imageRect = inputImageRects[0];

    // Edge modes other than 'none' do not inflate the affected paint rect.
    if (m_edgeMode != EdgeModeType::None)
        return enclosingIntRect(imageRect);

    auto kernelSize = calculateUnscaledKernelSize(filter.resolvedSize({ m_stdX, m_stdY }));

    // We take the half kernel size and multiply it with three, because we run box blur three times.
    imageRect.inflateX(3 * kernelSize.width() * 0.5f);
    imageRect.inflateY(3 * kernelSize.height() * 0.5f);

    return filter.clipToMaxEffectRect(imageRect, primitiveSubregion);
}

IntOutsets FEGaussianBlur::calculateOutsets(const FloatSize& stdDeviation)
{
    IntSize outsetSize = calculateOutsetSize(stdDeviation);
    return { outsetSize.height(), outsetSize.width(), outsetSize.height(), outsetSize.width() };
}

bool FEGaussianBlur::resultIsAlphaImage(const FilterImageVector& inputs) const
{
    return inputs[0]->isAlphaImage();
}

OptionSet<FilterRenderingMode> FEGaussianBlur::supportedFilterRenderingModes() const
{
    OptionSet<FilterRenderingMode> modes = FilterRenderingMode::Software;
#if USE(SKIA)
    if (m_edgeMode == EdgeModeType::None)
        modes.add(FilterRenderingMode::Accelerated);
#endif
    // FIXME: Ensure the correctness of the CG GaussianBlur filter (http://webkit.org/b/243816).
#if 0 && HAVE(CGSTYLE_COLORMATRIX_BLUR)
    if (m_stdX == m_stdY)
        modes.add(FilterRenderingMode::GraphicsContext);
#endif
    return modes;
}

std::unique_ptr<FilterEffectApplier> FEGaussianBlur::createAcceleratedApplier() const
{
#if USE(SKIA)
    return FilterEffectApplier::create<FEGaussianBlurSkiaApplier>(*this);
#else
    return nullptr;
#endif
}

std::unique_ptr<FilterEffectApplier> FEGaussianBlur::createSoftwareApplier() const
{
#if USE(SKIA)
    if (m_edgeMode == EdgeModeType::None)
        return FilterEffectApplier::create<FEGaussianBlurSkiaApplier>(*this);
#endif
    return FilterEffectApplier::create<FEGaussianBlurSoftwareApplier>(*this);
}

std::optional<GraphicsStyle> FEGaussianBlur::createGraphicsStyle(GraphicsContext&, const Filter& filter) const
{
    auto radius = calculateUnscaledKernelSize(filter.resolvedSize({ m_stdX, m_stdY }));
    return GraphicsGaussianBlur { radius };
}

TextStream& FEGaussianBlur::externalRepresentation(TextStream& ts, FilterRepresentation representation) const
{
    ts << indent << "[feGaussianBlur";
    FilterEffect::externalRepresentation(ts, representation);

    ts << " stdDeviation=\"" << m_stdX << ", " << m_stdY << "\"";

    ts << "]\n";
    return ts;
}

} // namespace WebCore
