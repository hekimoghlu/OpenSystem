/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 10, 2023.
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
**********************************************************************
* Copyright (C) 1998-2016, International Business Machines Corporation 
* and others.  All Rights Reserved.
**********************************************************************
*/
/***********************************************************************
*   Date        Name        Description
*   12/14/99    Madhu        Creation.
***********************************************************************/
/**
 * IntlTestRBBI is the medium level test class for RuleBasedBreakIterator
 */

#include "unicode/utypes.h"

#if !UCONFIG_NO_BREAK_ITERATION && !UCONFIG_NO_REGULAR_EXPRESSIONS

#include "intltest.h"
#include "itrbbi.h"
#include "lstmbetst.h"
#include "rbbiapts.h"
#include "rbbitst.h"
#include "rbbimonkeytest.h"


void IntlTestRBBI::runIndexedTest( int32_t index, UBool exec, const char* &name, char* par )
{
    if (exec) {
        logln("TestSuite RuleBasedBreakIterator: ");
    }
    TESTCASE_AUTO_BEGIN;
    TESTCASE_AUTO_CLASS(RBBIAPITest);
    TESTCASE_AUTO_CLASS(RBBITest);
#if !UCONFIG_NO_FORMATTING
    TESTCASE_AUTO_CLASS(RBBIMonkeyTest);
#endif
    TESTCASE_AUTO_CLASS(LSTMBETest);
    TESTCASE_AUTO_END;
}

#endif /* #if !UCONFIG_NO_BREAK_ITERATION && !UCONFIG_NO_REGULAR_EXPRESSIONS */
