/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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

// Â© 2018 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING
#ifndef __SOURCE_NUMBER_MULTIPLIER_H__
#define __SOURCE_NUMBER_MULTIPLIER_H__

#include "numparse_types.h"
#include "number_decimfmtprops.h"

U_NAMESPACE_BEGIN
namespace number::impl {

/**
 * Wraps a {@link Multiplier} for use in the number formatting pipeline.
 */
// Exported as U_I18N_API for tests
class U_I18N_API MultiplierFormatHandler : public MicroPropsGenerator, public UMemory {
  public:
    MultiplierFormatHandler() = default; // WARNING: Leaves object in an unusable state; call setAndChain()

    void setAndChain(const Scale& multiplier, const MicroPropsGenerator* parent);

    void processQuantity(DecimalQuantity& quantity, MicroProps& micros,
                         UErrorCode& status) const override;

  private:
    Scale fMultiplier;
    const MicroPropsGenerator *fParent;
};


/** Gets a Scale from a DecimalFormatProperties. In Java, defined in RoundingUtils.java */
static inline Scale scaleFromProperties(const DecimalFormatProperties& properties) {
    int32_t magnitudeMultiplier = properties.magnitudeMultiplier + properties.multiplierScale;
    int32_t arbitraryMultiplier = properties.multiplier;
    if (magnitudeMultiplier != 0 && arbitraryMultiplier != 1) {
        return Scale::byDoubleAndPowerOfTen(arbitraryMultiplier, magnitudeMultiplier);
    } else if (magnitudeMultiplier != 0) {
        return Scale::powerOfTen(magnitudeMultiplier);
    } else if (arbitraryMultiplier != 1) {
        return Scale::byDouble(arbitraryMultiplier);
    } else {
        return Scale::none();
    }
}

} // namespace number::impl
U_NAMESPACE_END

#endif //__SOURCE_NUMBER_MULTIPLIER_H__
#endif /* #if !UCONFIG_NO_FORMATTING */
