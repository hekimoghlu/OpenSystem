/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 31, 2024.
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
 * Copyright (c) 1997-2014, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/
/********************************************************************************
*
* File CGENDTST.C
*********************************************************************************
*/

/* C API TEST FOR GENDER INFO */

#include "unicode/utypes.h"
#include "cmemory.h"

#if !UCONFIG_NO_FORMATTING

#include "cintltst.h"
#include "unicode/ugender.h"

static const UGender kAllFemale[] = {UGENDER_FEMALE, UGENDER_FEMALE};

void addGendInfoForTest(TestNode** root);
static void TestGenderInfo(void);

#define TESTCASE(x) addTest(root, &x, "tsformat/cgendtst/" #x)

void addGendInfoForTest(TestNode** root)
{
    TESTCASE(TestGenderInfo);
}

static void TestGenderInfo(void) {
  UErrorCode status = U_ZERO_ERROR;
  const UGenderInfo* actual_gi = ugender_getInstance("fr_CA", &status);
  UGender actual;
  if (U_FAILURE(status)) {
    log_err_status(status, "Fail to create UGenderInfo - %s (Are you missing data?)", u_errorName(status));
    return;
  }
  actual = ugender_getListGender(actual_gi, kAllFemale, UPRV_LENGTHOF(kAllFemale), &status);
  if (U_FAILURE(status)) {
    log_err("Fail to get gender of list - %s\n", u_errorName(status));
    return;
  }
  if (actual != UGENDER_FEMALE) {
    log_err("Expected UGENDER_FEMALE got %d\n", actual);
  }
}

#endif /* #if !UCONFIG_NO_FORMATTING */
