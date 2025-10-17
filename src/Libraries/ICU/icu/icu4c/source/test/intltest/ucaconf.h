/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 28, 2022.
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
/********************************************************************
 * COPYRIGHT:
 * Copyright (c) 2002-2012, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/

/**
 * UCAConformanceTest performs conformance tests defined in the data
 * files. ICU ships with stub data files, as the whole test are too 
 * long. To do the whole test, download the test files.
 */

#ifndef _UCACONF_TST
#define _UCACONF_TST

#include "unicode/utypes.h"

#if !UCONFIG_NO_COLLATION

#include "unicode/tblcoll.h"
#include "tscoll.h"

#include <stdio.h>

class UCAConformanceTest: public IntlTestCollator {
public:
  UCAConformanceTest();
  virtual ~UCAConformanceTest();

  void runIndexedTest( int32_t index, UBool exec, const char* &name, char* /*par = nullptr */) override;

  void TestTableNonIgnorable(/* par */);
  void TestTableShifted(/* par */);     
  void TestRulesNonIgnorable(/* par */);
  void TestRulesShifted(/* par */);     
private:
  void initRbUCA();
  void setCollNonIgnorable(Collator *coll);
  void setCollShifted(Collator *coll);
  void testConformance(const Collator *coll);
  void openTestFile(const char *type);

  RuleBasedCollator *UCA;  // rule-based so rules are available
  Collator *rbUCA;
  FILE *testFile;
  UErrorCode status;
  char testDataPath[1024];
  UBool isAtLeastUCA62;
};

#endif /* #if !UCONFIG_NO_COLLATION */

#endif
