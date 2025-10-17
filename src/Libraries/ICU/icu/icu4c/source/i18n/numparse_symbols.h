/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 10, 2023.
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
#ifndef __NUMPARSE_SYMBOLS_H__
#define __NUMPARSE_SYMBOLS_H__

#include "numparse_types.h"
#include "unicode/uniset.h"
#include "static_unicode_sets.h"

U_NAMESPACE_BEGIN
namespace numparse::impl {

/**
 * A base class for many matchers that performs a simple match against a UnicodeString and/or UnicodeSet.
 *
 * @author sffc
 */
// Exported as U_I18N_API for tests
class U_I18N_API SymbolMatcher : public NumberParseMatcher, public UMemory {
  public:
    SymbolMatcher() = default;  // WARNING: Leaves the object in an unusable state

    const UnicodeSet* getSet() const;

    bool match(StringSegment& segment, ParsedNumber& result, UErrorCode& status) const override;

    bool smokeTest(const StringSegment& segment) const override;

    UnicodeString toString() const override;

    virtual bool isDisabled(const ParsedNumber& result) const = 0;

    virtual void accept(StringSegment& segment, ParsedNumber& result) const = 0;

  protected:
    UnicodeString fString;
    const UnicodeSet* fUniSet; // a reference from numparse_unisets.h; never owned

    SymbolMatcher(const UnicodeString& symbolString, unisets::Key key);
};


// Exported as U_I18N_API for tests
class U_I18N_API IgnorablesMatcher : public SymbolMatcher {
  public:
    IgnorablesMatcher() = default;  // WARNING: Leaves the object in an unusable state

    IgnorablesMatcher(parse_flags_t parseFlags);

    bool isFlexible() const override;

    UnicodeString toString() const override;

  protected:
    bool isDisabled(const ParsedNumber& result) const override;

    void accept(StringSegment& segment, ParsedNumber& result) const override;
};


class InfinityMatcher : public SymbolMatcher {
  public:
    InfinityMatcher() = default;  // WARNING: Leaves the object in an unusable state

    InfinityMatcher(const DecimalFormatSymbols& dfs);

  protected:
    bool isDisabled(const ParsedNumber& result) const override;

    void accept(StringSegment& segment, ParsedNumber& result) const override;
};


// Exported as U_I18N_API for tests
class U_I18N_API MinusSignMatcher : public SymbolMatcher {
  public:
    MinusSignMatcher() = default;  // WARNING: Leaves the object in an unusable state

    MinusSignMatcher(const DecimalFormatSymbols& dfs, bool allowTrailing);

  protected:
    bool isDisabled(const ParsedNumber& result) const override;

    void accept(StringSegment& segment, ParsedNumber& result) const override;

  private:
    bool fAllowTrailing;
};


class NanMatcher : public SymbolMatcher {
  public:
    NanMatcher() = default;  // WARNING: Leaves the object in an unusable state

    NanMatcher(const DecimalFormatSymbols& dfs);

  protected:
    bool isDisabled(const ParsedNumber& result) const override;

    void accept(StringSegment& segment, ParsedNumber& result) const override;
};


class PaddingMatcher : public SymbolMatcher {
  public:
    PaddingMatcher() = default;  // WARNING: Leaves the object in an unusable state

    PaddingMatcher(const UnicodeString& padString);

    bool isFlexible() const override;

  protected:
    bool isDisabled(const ParsedNumber& result) const override;

    void accept(StringSegment& segment, ParsedNumber& result) const override;
};


// Exported as U_I18N_API for tests
class U_I18N_API PercentMatcher : public SymbolMatcher {
  public:
    PercentMatcher() = default;  // WARNING: Leaves the object in an unusable state

    PercentMatcher(const DecimalFormatSymbols& dfs);

  protected:
    bool isDisabled(const ParsedNumber& result) const override;

    void accept(StringSegment& segment, ParsedNumber& result) const override;
};

// Exported as U_I18N_API for tests
class U_I18N_API PermilleMatcher : public SymbolMatcher {
  public:
    PermilleMatcher() = default;  // WARNING: Leaves the object in an unusable state

    PermilleMatcher(const DecimalFormatSymbols& dfs);

  protected:
    bool isDisabled(const ParsedNumber& result) const override;

    void accept(StringSegment& segment, ParsedNumber& result) const override;
};


// Exported as U_I18N_API for tests
class U_I18N_API PlusSignMatcher : public SymbolMatcher {
  public:
    PlusSignMatcher() = default;  // WARNING: Leaves the object in an unusable state

    PlusSignMatcher(const DecimalFormatSymbols& dfs, bool allowTrailing);

  protected:
    bool isDisabled(const ParsedNumber& result) const override;

    void accept(StringSegment& segment, ParsedNumber& result) const override;

  private:
    bool fAllowTrailing;
};

} // namespace numparse::impl
U_NAMESPACE_END

#endif //__NUMPARSE_SYMBOLS_H__
#endif /* #if !UCONFIG_NO_FORMATTING */
