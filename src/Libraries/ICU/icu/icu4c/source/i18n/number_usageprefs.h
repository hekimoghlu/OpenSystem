/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 18, 2024.
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

// Â© 2020 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING
#ifndef __NUMBER_USAGEPREFS_H__
#define __NUMBER_USAGEPREFS_H__

#include "cmemory.h"
#include "number_types.h"
#include "unicode/listformatter.h"
#include "unicode/localpointer.h"
#include "unicode/locid.h"
#include "unicode/measunit.h"
#include "unicode/stringpiece.h"
#include "unicode/uobject.h"
#include "units_converter.h"
#include "units_router.h"

U_NAMESPACE_BEGIN

using ::icu::units::ComplexUnitsConverter;
using ::icu::units::UnitsRouter;

namespace number::impl {

/**
 * A MicroPropsGenerator which uses UnitsRouter to produce output converted to a
 * MeasureUnit appropriate for a particular localized usage: see
 * NumberFormatterSettings::usage().
 */
class U_I18N_API UsagePrefsHandler : public MicroPropsGenerator, public UMemory {
  public:
    UsagePrefsHandler(const Locale &locale, const MeasureUnit &inputUnit, const StringPiece usage,
                      const MicroPropsGenerator *parent, UErrorCode &status);

    /**
     * Obtains the appropriate output value, MeasureUnit and
     * rounding/precision behaviour from the UnitsRouter.
     *
     * The output unit is passed on to the LongNameHandler via
     * micros.outputUnit.
     */
    void processQuantity(DecimalQuantity &quantity, MicroProps &micros,
                         UErrorCode &status) const override;

    /**
     * Returns the list of possible output units, i.e. the full set of
     * preferences, for the localized, usage-specific unit preferences.
     *
     * The returned pointer should be valid for the lifetime of the
     * UsagePrefsHandler instance.
     */
    const MaybeStackVector<MeasureUnit> *getOutputUnits() const {
        return fUnitsRouter.getOutputUnits();
    }

  private:
    UnitsRouter fUnitsRouter;
    const MicroPropsGenerator *fParent;
};

} // namespace number::impl

// Export explicit template instantiations of LocalPointerBase and LocalPointer.
// This is required when building DLLs for Windows. (See datefmt.h,
// collationiterator.h, erarules.h and others for similar examples.)
//
// Note: These need to be outside of the number::impl namespace, or Clang will
// generate a compile error.
#if U_PF_WINDOWS <= U_PLATFORM && U_PLATFORM <= U_PF_CYGWIN
template class U_I18N_API LocalPointerBase<ComplexUnitsConverter>;
template class U_I18N_API LocalPointer<ComplexUnitsConverter>;
#endif

namespace number::impl {

/**
 * A MicroPropsGenerator which converts a measurement from one MeasureUnit to
 * another. In particular, the output MeasureUnit may be a mixed unit. (The
 * input unit may not be a mixed unit.)
 */
class U_I18N_API UnitConversionHandler : public MicroPropsGenerator, public UMemory {
  public:
    /**
     * Constructor.
     *
     * @param targetUnit Specifies the output MeasureUnit. The input MeasureUnit
     *     is derived from it: in case of a mixed unit, the biggest unit is
     *     taken as the input unit. If not a mixed unit, the input unit will be
     *     the same as the output unit and no unit conversion takes place.
     * @param parent The parent MicroPropsGenerator.
     * @param status Receives status.
     */
    UnitConversionHandler(const MeasureUnit &targetUnit, const MicroPropsGenerator *parent,
                          UErrorCode &status);

    /**
     * Obtains the appropriate output values from the Unit Converter.
     */
    void processQuantity(DecimalQuantity &quantity, MicroProps &micros,
                         UErrorCode &status) const override;
  private:
    MeasureUnit fOutputUnit;
    LocalPointer<ComplexUnitsConverter> fUnitConverter;
    const MicroPropsGenerator *fParent;
};

} // namespace number::impl

U_NAMESPACE_END

#endif // __NUMBER_USAGEPREFS_H__
#endif /* #if !UCONFIG_NO_FORMATTING */
