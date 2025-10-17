/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 23, 2024.
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
#ifndef __SOURCE_NUMPARSE_VALIDATORS_H__
#define __SOURCE_NUMPARSE_VALIDATORS_H__

#include "numparse_types.h"
#include "static_unicode_sets.h"

U_NAMESPACE_BEGIN
namespace numparse::impl {

class ValidationMatcher : public NumberParseMatcher {
  public:
    bool match(StringSegment&, ParsedNumber&, UErrorCode&) const override {
        // No-op
        return false;
    }

    bool smokeTest(const StringSegment&) const override {
        // No-op
        return false;
    }

    void postProcess(ParsedNumber& result) const override = 0;
};


class RequireAffixValidator : public ValidationMatcher, public UMemory {
  public:
    void postProcess(ParsedNumber& result) const override;

    UnicodeString toString() const override;
};


class RequireCurrencyValidator : public ValidationMatcher, public UMemory {
  public:
    void postProcess(ParsedNumber& result) const override;

    UnicodeString toString() const override;
};


class RequireDecimalSeparatorValidator : public ValidationMatcher, public UMemory {
  public:
    RequireDecimalSeparatorValidator() = default;  // leaves instance in valid but undefined state

    RequireDecimalSeparatorValidator(bool patternHasDecimalSeparator);

    void postProcess(ParsedNumber& result) const override;

    UnicodeString toString() const override;

  private:
    bool fPatternHasDecimalSeparator;
};


class RequireNumberValidator : public ValidationMatcher, public UMemory {
  public:
    void postProcess(ParsedNumber& result) const override;

    UnicodeString toString() const override;
};


/**
 * Wraps a {@link Multiplier} for use in the number parsing pipeline.
 */
class MultiplierParseHandler : public ValidationMatcher, public UMemory {
  public:
    MultiplierParseHandler() = default;  // leaves instance in valid but undefined state

    MultiplierParseHandler(::icu::number::Scale multiplier);

    void postProcess(ParsedNumber& result) const override;

    UnicodeString toString() const override;

  private:
    ::icu::number::Scale fMultiplier;
};

} // namespace numparse::impl
U_NAMESPACE_END

#endif //__SOURCE_NUMPARSE_VALIDATORS_H__
#endif /* #if !UCONFIG_NO_FORMATTING */
