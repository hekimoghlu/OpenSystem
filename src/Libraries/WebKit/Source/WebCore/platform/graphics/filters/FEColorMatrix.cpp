/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 4, 2023.
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
#include "FEColorMatrix.h"

#include "ColorMatrix.h"
#include "FEColorMatrixSoftwareApplier.h"
#include "Filter.h"
#include <wtf/text/TextStream.h>

#if USE(CORE_IMAGE)
#include "FEColorMatrixCoreImageApplier.h"
#endif

#if USE(SKIA)
#include "FEColorMatrixSkiaApplier.h"
#endif

namespace WebCore {

Ref<FEColorMatrix> FEColorMatrix::create(ColorMatrixType type, Vector<float>&& values, DestinationColorSpace colorSpace)
{
    return adoptRef(*new FEColorMatrix(type, WTFMove(values), colorSpace));
}

FEColorMatrix::FEColorMatrix(ColorMatrixType type, Vector<float>&& values, DestinationColorSpace colorSpace)
    : FilterEffect(FilterEffect::Type::FEColorMatrix, colorSpace)
    , m_type(type)
    , m_values(WTFMove(values))
{
}

bool FEColorMatrix::operator==(const FEColorMatrix& other) const
{
    return FilterEffect::operator==(other)
        && m_type == other.m_type
        && m_values == other.m_values;
}

bool FEColorMatrix::setType(ColorMatrixType type)
{
    if (m_type == type)
        return false;
    m_type = type;
    return true;
}

bool FEColorMatrix::setValues(const Vector<float> &values)
{
    if (m_values == values)
        return false;
    m_values = values;
    return true;
}

void FEColorMatrix::calculateSaturateComponents(std::span<float, 9> components, float value)
{
    auto saturationMatrix = saturationColorMatrix(value);

    components[0] = saturationMatrix.at(0, 0);
    components[1] = saturationMatrix.at(0, 1);
    components[2] = saturationMatrix.at(0, 2);

    components[3] = saturationMatrix.at(1, 0);
    components[4] = saturationMatrix.at(1, 1);
    components[5] = saturationMatrix.at(1, 2);

    components[6] = saturationMatrix.at(2, 0);
    components[7] = saturationMatrix.at(2, 1);
    components[8] = saturationMatrix.at(2, 2);
}

void FEColorMatrix::calculateHueRotateComponents(std::span<float, 9> components, float angleInDegrees)
{
    auto hueRotateMatrix = hueRotateColorMatrix(angleInDegrees);

    components[0] = hueRotateMatrix.at(0, 0);
    components[1] = hueRotateMatrix.at(0, 1);
    components[2] = hueRotateMatrix.at(0, 2);

    components[3] = hueRotateMatrix.at(1, 0);
    components[4] = hueRotateMatrix.at(1, 1);
    components[5] = hueRotateMatrix.at(1, 2);

    components[6] = hueRotateMatrix.at(2, 0);
    components[7] = hueRotateMatrix.at(2, 1);
    components[8] = hueRotateMatrix.at(2, 2);
}

Vector<float> FEColorMatrix::normalizedFloats(const Vector<float>& values)
{
    Vector<float> normalizedValues(values.size());
    for (size_t i = 0; i < values.size(); ++i)
        normalizedValues[i] = normalizedFloat(values[i]);
    return normalizedValues;
}

bool FEColorMatrix::resultIsAlphaImage(const FilterImageVector&) const
{
    return m_type == ColorMatrixType::FECOLORMATRIX_TYPE_LUMINANCETOALPHA;
}

OptionSet<FilterRenderingMode> FEColorMatrix::supportedFilterRenderingModes() const
{
    OptionSet<FilterRenderingMode> modes = FilterRenderingMode::Software;
#if USE(CORE_IMAGE)
    if (FEColorMatrixCoreImageApplier::supportsCoreImageRendering(*this))
        modes.add(FilterRenderingMode::Accelerated);
#endif
#if USE(SKIA)
    modes.add(FilterRenderingMode::Accelerated);
#endif
#if HAVE(CGSTYLE_COLORMATRIX_BLUR)
    if (m_type == ColorMatrixType::FECOLORMATRIX_TYPE_MATRIX)
        modes.add(FilterRenderingMode::GraphicsContext);
#endif
    return modes;
}

std::unique_ptr<FilterEffectApplier> FEColorMatrix::createAcceleratedApplier() const
{
#if USE(CORE_IMAGE)
    return FilterEffectApplier::create<FEColorMatrixCoreImageApplier>(*this);
#elif USE(SKIA)
    return FilterEffectApplier::create<FEColorMatrixSkiaApplier>(*this);
#else
    return nullptr;
#endif
}

std::unique_ptr<FilterEffectApplier> FEColorMatrix::createSoftwareApplier() const
{
#if USE(SKIA)
    return FilterEffectApplier::create<FEColorMatrixSkiaApplier>(*this);
#else
    return FilterEffectApplier::create<FEColorMatrixSoftwareApplier>(*this);
#endif
}

std::optional<GraphicsStyle> FEColorMatrix::createGraphicsStyle(GraphicsContext&, const Filter&) const
{
    std::array<float, 20> values;
    std::copy_n(m_values.begin(), std::min<size_t>(m_values.size(), 20), values.begin());
    return GraphicsColorMatrix { values };
}

static TextStream& operator<<(TextStream& ts, const ColorMatrixType& type)
{
    switch (type) {
    case ColorMatrixType::FECOLORMATRIX_TYPE_UNKNOWN:
        ts << "UNKNOWN";
        break;
    case ColorMatrixType::FECOLORMATRIX_TYPE_MATRIX:
        ts << "MATRIX";
        break;
    case ColorMatrixType::FECOLORMATRIX_TYPE_SATURATE:
        ts << "SATURATE";
        break;
    case ColorMatrixType::FECOLORMATRIX_TYPE_HUEROTATE:
        ts << "HUEROTATE";
        break;
    case ColorMatrixType::FECOLORMATRIX_TYPE_LUMINANCETOALPHA:
        ts << "LUMINANCETOALPHA";
        break;
    }
    return ts;
}

TextStream& FEColorMatrix::externalRepresentation(TextStream& ts, FilterRepresentation representation) const
{
    ts << indent << "[feColorMatrix";
    FilterEffect::externalRepresentation(ts, representation);

    ts << " type=\"" << m_type << "\"";
    if (!m_values.isEmpty()) {
        ts << " values=\"";
        bool isFirst = true;
        for (auto value : m_values) {
            if (isFirst)
                isFirst = false;
            else
                ts << " "_s;
            ts << value;
        }
        ts << "\"";
    }

    ts << "]\n";
    return ts;
}

} // namespace WebCore
