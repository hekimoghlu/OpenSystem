/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 24, 2022.
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

// Â© 2016 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html
/*
 *******************************************************************************
 * Copyright (C) 1996-2015, International Business Machines Corporation and    *
 * others. All Rights Reserved.                                                *
 *******************************************************************************
 */

#ifndef ITRBNF_H
#define ITRBNF_H

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "intltest.h"
#include "unicode/rbnf.h"


class IntlTestRBNF : public IntlTest {
 public:

  // IntlTest override
  virtual void runIndexedTest(int32_t index, UBool exec, const char* &name, char* par) override;

#if U_HAVE_RBNF
  /** 
   * Perform an API test
   */
  virtual void TestAPI();

  void TestMultiplePluralRules();

  /**
   * Perform a simple spot check on the FractionalRuleSet logic
   */
  virtual void TestFractionalRuleSet();

#if 0
  /**
   * Perform API tests on llong
   */
  virtual void TestLLong();
  virtual void TestLLongConstructors();
  virtual void TestLLongSimpleOperators();
#endif

  /**
   * Perform a simple spot check on the English spellout rules
   */
  void TestEnglishSpellout();

  /**
   * Perform a simple spot check on the English ordinal-abbreviation rules
   */
  void TestOrdinalAbbreviations();

  /**
   * Perform a simple spot check on the duration-formatting rules
   */
  void TestDurations();
    
  /**
   * Test that rounding works correctly on multiplier substitutions that use
   * a DecimalFormat.
   */
  void TestDFRounding();

  /**
   * Perform a simple spot check on the Spanish spellout rules
   */
  void TestSpanishSpellout();

  /**
   * Perform a simple spot check on the French spellout rules
   */
  void TestFrenchSpellout();

  /**
   * Perform a simple spot check on the Swiss French spellout rules
   */
  void TestSwissFrenchSpellout();

  /**
   * Check that Belgian French matches Swiss French spellout rules
   */
  void TestBelgianFrenchSpellout();

  /**
   * Perform a simple spot check on the Italian spellout rules
   */
  void TestItalianSpellout();

  /**
   * Perform a simple spot check on the Portuguese spellout rules
   */
  void TestPortugueseSpellout();

  /**
   * Perform a simple spot check on the German spellout rules
   */
  void TestGermanSpellout();

  /**
   * Perform a simple spot check on the Thai spellout rules
   */
  void TestThaiSpellout();

  /**
   * Perform a simple spot check on the Norwegian (no,nb) spellout rules
   */
  void TestNorwegianSpellout();

  /**
   * Perform a simple spot check on the Swedish spellout rules
   */
  void TestSwedishSpellout();

  /**
   * Perform a simple spot check on the Gujarati spellout rules
   */
  void TestGujaratiSpellout();
  
  /**
   * Perform a simple spot check on small values
   */
  void TestSmallValues();

  /**
   * Test localizations using string data.
   */
  void TestLocalizations();

  /**
   * Test that all locales construct ok.
   */
  void TestAllLocales();

  /**
   * Test that hebrew fractions format without trailing '<'
   */
  void TestHebrewFraction();

  /**
   * Regression test, don't truncate
   * when doing multiplier substitution to a number format rule.
   */
  void TestMultiplierSubstitution();

  /**
   * Test the setDecimalFormatSymbols in RBNF
   */
  void TestSetDecimalFormatSymbols();

  /**
   * Test the plural rules in RBNF
   */
  void TestPluralRules();

    void TestInfinityNaN();
    void TestVariableDecimalPoint();
    void TestRounding();
    void TestLargeNumbers();
    void TestCompactDecimalFormatStyle();
    void TestParseFailure();
    void TestMinMaxIntegerDigitsIgnored();
    void TestNumberingSystem();
    void TestMemoryLeak22899();
    void TestInfiniteRecursion();

protected:
  virtual void doTest(RuleBasedNumberFormat* formatter, const char* const testData[][2], UBool testParsing);
  virtual void doLenientParseTest(RuleBasedNumberFormat* formatter, const char* testData[][2]);

/* U_HAVE_RBNF */
#else

  virtual void TestRBNFDisabled();

/* U_HAVE_RBNF */
#endif
};

#endif /* #if !UCONFIG_NO_FORMATTING */

// endif ITRBNF_H
#endif
