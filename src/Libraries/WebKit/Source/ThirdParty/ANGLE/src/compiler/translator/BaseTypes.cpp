/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 11, 2024.
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

//
// Copyright 2022 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "BaseTypes.h"

#include "common/PackedEnums.h"

namespace sh
{
namespace
{
constexpr gl::BlendEquationBitSet kAdvancedBlendBits{
    gl::BlendEquationType::Multiply,      gl::BlendEquationType::Screen,
    gl::BlendEquationType::Overlay,       gl::BlendEquationType::Darken,
    gl::BlendEquationType::Lighten,       gl::BlendEquationType::Colordodge,
    gl::BlendEquationType::Colorburn,     gl::BlendEquationType::Hardlight,
    gl::BlendEquationType::Softlight,     gl::BlendEquationType::Difference,
    gl::BlendEquationType::Exclusion,     gl::BlendEquationType::HslHue,
    gl::BlendEquationType::HslSaturation, gl::BlendEquationType::HslColor,
    gl::BlendEquationType::HslLuminosity,
};

constexpr gl::BlendEquationBitSet kAdvancedBlendHslBits{
    gl::BlendEquationType::HslHue,
    gl::BlendEquationType::HslSaturation,
    gl::BlendEquationType::HslColor,
    gl::BlendEquationType::HslLuminosity,
};

bool IsValidAdvancedBlendBitSet(uint32_t enabledEquations)
{
    return (gl::BlendEquationBitSet(enabledEquations) & ~kAdvancedBlendBits).none();
}
}  // anonymous namespace

bool AdvancedBlendEquations::any() const
{
    ASSERT(IsValidAdvancedBlendBitSet(mEnabledBlendEquations));
    return mEnabledBlendEquations != 0;
}

bool AdvancedBlendEquations::all() const
{
    ASSERT(IsValidAdvancedBlendBitSet(mEnabledBlendEquations));
    return gl::BlendEquationBitSet(mEnabledBlendEquations) == kAdvancedBlendBits;
}

bool AdvancedBlendEquations::anyHsl() const
{
    ASSERT(IsValidAdvancedBlendBitSet(mEnabledBlendEquations));
    return (gl::BlendEquationBitSet(mEnabledBlendEquations) & kAdvancedBlendHslBits).any();
}

void AdvancedBlendEquations::setAll()
{
    ASSERT(IsValidAdvancedBlendBitSet(mEnabledBlendEquations));
    mEnabledBlendEquations = kAdvancedBlendBits.bits();
}

void AdvancedBlendEquations::set(uint32_t blendEquation)
{
    gl::BlendEquationType eq = static_cast<gl::BlendEquationType>(blendEquation);
    mEnabledBlendEquations   = gl::BlendEquationBitSet(mEnabledBlendEquations).set(eq).bits();
    ASSERT(IsValidAdvancedBlendBitSet(mEnabledBlendEquations));
}

const char *AdvancedBlendEquations::GetLayoutString(uint32_t blendEquation)
{
    switch (static_cast<gl::BlendEquationType>(blendEquation))
    {
        case gl::BlendEquationType::Multiply:
            return "blend_support_multiply";
        case gl::BlendEquationType::Screen:
            return "blend_support_screen";
        case gl::BlendEquationType::Overlay:
            return "blend_support_overlay";
        case gl::BlendEquationType::Darken:
            return "blend_support_darken";
        case gl::BlendEquationType::Lighten:
            return "blend_support_lighten";
        case gl::BlendEquationType::Colordodge:
            return "blend_support_colordodge";
        case gl::BlendEquationType::Colorburn:
            return "blend_support_colorburn";
        case gl::BlendEquationType::Hardlight:
            return "blend_support_hardlight";
        case gl::BlendEquationType::Softlight:
            return "blend_support_softlight";
        case gl::BlendEquationType::Difference:
            return "blend_support_difference";
        case gl::BlendEquationType::Exclusion:
            return "blend_support_exclusion";
        case gl::BlendEquationType::HslHue:
            return "blend_support_hsl_hue";
        case gl::BlendEquationType::HslSaturation:
            return "blend_support_hsl_saturation";
        case gl::BlendEquationType::HslColor:
            return "blend_support_hsl_color";
        case gl::BlendEquationType::HslLuminosity:
            return "blend_support_hsl_luminosity";
        default:
            UNREACHABLE();
            return nullptr;
    }
}

const char *AdvancedBlendEquations::GetAllEquationsLayoutString()
{
    return "blend_support_all_equations";
}

}  // namespace sh
