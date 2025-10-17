/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 1, 2023.
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

#include "FilterEffect.h"
#include <wtf/EnumeratedArray.h>
#include <wtf/Vector.h>

namespace WebCore {

enum class ComponentTransferType : uint8_t {
    FECOMPONENTTRANSFER_TYPE_UNKNOWN  = 0,
    FECOMPONENTTRANSFER_TYPE_IDENTITY = 1,
    FECOMPONENTTRANSFER_TYPE_TABLE    = 2,
    FECOMPONENTTRANSFER_TYPE_DISCRETE = 3,
    FECOMPONENTTRANSFER_TYPE_LINEAR   = 4,
    FECOMPONENTTRANSFER_TYPE_GAMMA    = 5
};

struct ComponentTransferFunction {
    ComponentTransferType type { ComponentTransferType::FECOMPONENTTRANSFER_TYPE_UNKNOWN };

    float slope { 0 };
    float intercept { 0 };
    float amplitude { 0 };
    float exponent { 0 };
    float offset { 0 };

    Vector<float> tableValues;

    bool operator==(const ComponentTransferFunction&) const = default;
};

enum class ComponentTransferChannel : uint8_t { Red, Green, Blue, Alpha };

} // namespace WebCore

namespace WebCore {

using ComponentTransferFunctions = EnumeratedArray<ComponentTransferChannel, ComponentTransferFunction, ComponentTransferChannel::Alpha>;

class FEComponentTransfer : public FilterEffect {
public:
    WEBCORE_EXPORT static Ref<FEComponentTransfer> create(const ComponentTransferFunction& redFunc, const ComponentTransferFunction& greenFunc, const ComponentTransferFunction& blueFunc, const ComponentTransferFunction& alphaFunc, DestinationColorSpace = DestinationColorSpace::SRGB());
    static Ref<FEComponentTransfer> create(ComponentTransferFunctions&&);

    using LookupTable = std::array<uint8_t, 256>;
    static LookupTable computeLookupTable(const ComponentTransferFunction&);

    bool operator==(const FEComponentTransfer&) const;

    ComponentTransferFunction redFunction() const { return m_functions[ComponentTransferChannel::Red]; }
    ComponentTransferFunction greenFunction() const { return m_functions[ComponentTransferChannel::Green]; }
    ComponentTransferFunction blueFunction() const { return m_functions[ComponentTransferChannel::Blue]; }
    ComponentTransferFunction alphaFunction() const { return m_functions[ComponentTransferChannel::Alpha]; }

    bool setType(ComponentTransferChannel, ComponentTransferType);
    bool setSlope(ComponentTransferChannel, float);
    bool setIntercept(ComponentTransferChannel, float);
    bool setAmplitude(ComponentTransferChannel, float);
    bool setExponent(ComponentTransferChannel, float);
    bool setOffset(ComponentTransferChannel, float);
    bool setTableValues(ComponentTransferChannel, Vector<float>&&);

private:
    FEComponentTransfer(const ComponentTransferFunction& redFunc, const ComponentTransferFunction& greenFunc, const ComponentTransferFunction& blueFunc, const ComponentTransferFunction& alphaFunc, DestinationColorSpace);
    FEComponentTransfer(ComponentTransferFunctions&&);

    bool operator==(const FilterEffect& other) const override { return areEqual<FEComponentTransfer>(*this, other); }

    OptionSet<FilterRenderingMode> supportedFilterRenderingModes() const override;
    std::unique_ptr<FilterEffectApplier> createAcceleratedApplier() const override;
    std::unique_ptr<FilterEffectApplier> createSoftwareApplier() const override;

    WTF::TextStream& externalRepresentation(WTF::TextStream&, FilterRepresentation) const override;

    ComponentTransferFunctions m_functions;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_FILTER_FUNCTION(FEComponentTransfer)
