/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 9, 2024.
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
#ifndef __NUMPARSE_SCIENTIFIC_H__
#define __NUMPARSE_SCIENTIFIC_H__

#include "numparse_types.h"
#include "numparse_decimal.h"
#include "numparse_symbols.h"
#include "unicode/numberformatter.h"

using icu::number::impl::Grouper;

U_NAMESPACE_BEGIN
namespace numparse::impl {

class ScientificMatcher : public NumberParseMatcher, public UMemory {
  public:
    ScientificMatcher() = default;  // WARNING: Leaves the object in an unusable state

    ScientificMatcher(const DecimalFormatSymbols& dfs, const Grouper& grouper);

    bool match(StringSegment& segment, ParsedNumber& result, UErrorCode& status) const override;

    bool smokeTest(const StringSegment& segment) const override;

    UnicodeString toString() const override;

  private:
    UnicodeString fExponentSeparatorString;
    DecimalMatcher fExponentMatcher;
    IgnorablesMatcher fIgnorablesMatcher;
    UnicodeString fCustomMinusSign;
    UnicodeString fCustomPlusSign;
};

} // namespace numparse::impl
U_NAMESPACE_END

#endif //__NUMPARSE_SCIENTIFIC_H__
#endif /* #if !UCONFIG_NO_FORMATTING */
