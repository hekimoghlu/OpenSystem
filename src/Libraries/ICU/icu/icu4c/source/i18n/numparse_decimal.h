/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 12, 2023.
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
#ifndef __NUMPARSE_DECIMAL_H__
#define __NUMPARSE_DECIMAL_H__

#include "unicode/uniset.h"
#include "numparse_types.h"

U_NAMESPACE_BEGIN
namespace numparse::impl {

using ::icu::number::impl::Grouper;

class DecimalMatcher : public NumberParseMatcher, public UMemory {
  public:
    DecimalMatcher() = default;  // WARNING: Leaves the object in an unusable state

    DecimalMatcher(const DecimalFormatSymbols& symbols, const Grouper& grouper,
                   parse_flags_t parseFlags);

    bool match(StringSegment& segment, ParsedNumber& result, UErrorCode& status) const override;

    bool
    match(StringSegment& segment, ParsedNumber& result, int8_t exponentSign, UErrorCode& status) const;

    bool smokeTest(const StringSegment& segment) const override;

    UnicodeString toString() const override;

  private:
    /** If true, only accept strings whose grouping sizes match the locale */
    bool requireGroupingMatch;

    /** If true, do not accept grouping separators at all */
    bool groupingDisabled;

    // Fraction grouping parsing is disabled for now but could be enabled later.
    // See https://unicode-org.atlassian.net/browse/ICU-10794
    // bool fractionGrouping;

    /** If true, do not accept numbers in the fraction */
    bool integerOnly;

    int16_t grouping1;
    int16_t grouping2;

    UnicodeString groupingSeparator;
    UnicodeString decimalSeparator;

    // Assumption: these sets all consist of single code points. If this assumption needs to be broken,
    // fix getLeadCodePoints() as well as matching logic. Be careful of the performance impact.
    const UnicodeSet* groupingUniSet;
    const UnicodeSet* decimalUniSet;
    const UnicodeSet* separatorSet;
    const UnicodeSet* leadSet;

    // Make this class the owner of a few objects that could be allocated.
    // The first three LocalPointers are used for assigning ownership only.
    LocalPointer<const UnicodeSet> fLocalDecimalUniSet;
    LocalPointer<const UnicodeSet> fLocalSeparatorSet;
    LocalArray<const UnicodeString> fLocalDigitStrings;

    bool validateGroup(int32_t sepType, int32_t count, bool isPrimary) const;
};

} // namespace numparse::impl
U_NAMESPACE_END

#endif //__NUMPARSE_DECIMAL_H__
#endif /* #if !UCONFIG_NO_FORMATTING */
