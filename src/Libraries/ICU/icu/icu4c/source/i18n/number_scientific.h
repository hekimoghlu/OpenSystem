/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 8, 2023.
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

// Â© 2017 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING
#ifndef __NUMBER_SCIENTIFIC_H__
#define __NUMBER_SCIENTIFIC_H__

#include "number_types.h"

U_NAMESPACE_BEGIN
namespace number::impl {

// Forward-declare
class ScientificHandler;

class U_I18N_API ScientificModifier : public UMemory, public Modifier {
  public:
    ScientificModifier();

    void set(int32_t exponent, const ScientificHandler *handler);

    int32_t apply(FormattedStringBuilder &output, int32_t leftIndex, int32_t rightIndex,
                  UErrorCode &status) const override;

    int32_t getPrefixLength() const override;

    int32_t getCodePointCount() const override;

    bool isStrong() const override;

    bool containsField(Field field) const override;

    void getParameters(Parameters& output) const override;

    bool strictEquals(const Modifier& other) const override;

  private:
    int32_t fExponent;
    const ScientificHandler *fHandler;
};

class ScientificHandler : public UMemory, public MicroPropsGenerator, public MultiplierProducer {
  public:
    ScientificHandler(const Notation *notation, const DecimalFormatSymbols *symbols,
                      const MicroPropsGenerator *parent);

    void
    processQuantity(DecimalQuantity &quantity, MicroProps &micros, UErrorCode &status) const override;

    int32_t getMultiplier(int32_t magnitude) const override;

  private:
    const Notation::ScientificSettings fSettings;
    const DecimalFormatSymbols *fSymbols;
    const MicroPropsGenerator *fParent;

    friend class ScientificModifier;
};

} // namespace number::impl
U_NAMESPACE_END

#endif //__NUMBER_SCIENTIFIC_H__

#endif /* #if !UCONFIG_NO_FORMATTING */
