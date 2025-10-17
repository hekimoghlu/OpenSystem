/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 18, 2023.
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
 * Copyright (C) 1996-2006, International Business Machines Corporation and    *
 * others. All Rights Reserved.                                                *
 *******************************************************************************
 */

#ifndef ITRBNFRT_H
#define ITRBNFRT_H

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "intltest.h"
#include "unicode/rbnf.h"

class RbnfRoundTripTest : public IntlTest {

  // IntlTest override
  virtual void runIndexedTest(int32_t index, UBool exec, const char* &name, char* par) override;

#if U_HAVE_RBNF
  /**
   * Perform an exhaustive round-trip test on the English spellout rules
   */
  virtual void TestEnglishSpelloutRT();

  /**
   * Perform an exhaustive round-trip test on the duration-formatting rules
   */
  virtual void TestDurationsRT();

  /**
   * Perform an exhaustive round-trip test on the Spanish spellout rules
   */
  virtual void TestSpanishSpelloutRT();

  /**
   * Perform an exhaustive round-trip test on the French spellout rules
   */
  virtual void TestFrenchSpelloutRT();

  /**
   * Perform an exhaustive round-trip test on the Swiss French spellout rules
   */
  virtual void TestSwissFrenchSpelloutRT();

  /**
   * Perform an exhaustive round-trip test on the Italian spellout rules
   */
  virtual void TestItalianSpelloutRT();

  /**
   * Perform an exhaustive round-trip test on the German spellout rules
   */
  virtual void TestGermanSpelloutRT();

  /**
   * Perform an exhaustive round-trip test on the Swedish spellout rules
   */
  virtual void TestSwedishSpelloutRT();

  /**
   * Perform an exhaustive round-trip test on the Dutch spellout rules
   */
  virtual void TestDutchSpelloutRT();

  /**
   * Perform an exhaustive round-trip test on the Japanese spellout rules
   */
  virtual void TestJapaneseSpelloutRT();

  /**
   * Perform an exhaustive round-trip test on the Russian spellout rules
   */
  virtual void TestRussianSpelloutRT();

  /**
   * Perform an exhaustive round-trip test on the Portuguese spellout rules
   */
  virtual void TestPortugueseSpelloutRT();
  
  /**
   * Perform an exhaustive round-trip test on the Gujarati spellout rules
   */
  virtual void TestGujaratiSpelloutRT();

 protected:
  void doTest(const RuleBasedNumberFormat* formatter,  double lowLimit, double highLimit);

  /* U_HAVE_RBNF */
#else

  void TestRBNFDisabled();

  /* U_HAVE_RBNF */
#endif
};

#endif /* #if !UCONFIG_NO_FORMATTING */

// endif ITRBNFRT_H
#endif
