/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 28, 2021.
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
/***************************************************************************
*
*   Copyright (C) 2000-2007, International Business Machines
*   Corporation and others.  All Rights Reserved.
*
************************************************************************
*   Date        Name        Description
*   01/03/2000  Madhu        Creation.
*   03/2000     Madhu        Added additional tests
***********************************************************************/
/**
 * IntlTestTransliterator is the medium level test class for Transliterator 
 */

#include "unicode/utypes.h"

#if !UCONFIG_NO_TRANSLITERATION

#include "ittrans.h"
#include "transtst.h"
#include "transapi.h"
#include "cpdtrtst.h"
#include "transrt.h"
#include "jamotest.h"
#include "trnserr.h"
#include "reptest.h"

#define CASE(id,test) case id:                                \
                          name = #test;                       \
                          if (exec) {                         \
                              logln(#test "---"); logln();    \
                              test t;                         \
                              callTest(t, par);               \
                          }                                   \
                          break

void IntlTestTransliterator::runIndexedTest( int32_t index, UBool exec, const char* &name, char* par )
{
    if (exec) logln("TestSuite Transliterator");
    switch (index) {
        CASE(0, TransliteratorTest);
        CASE(1, TransliteratorAPITest);
        CASE(2, CompoundTransliteratorTest);
        CASE(3, TransliteratorRoundTripTest);
        CASE(4, JamoTest);
        CASE(5, TransliteratorErrorTest);
        CASE(6, ReplaceableTest);
#if !UCONFIG_NO_TRANSLITERATION && defined(U_USE_UNICODE_FILTER_LOGIC_OBSOLETE_2_8)
        CASE(7, UnicodeFilterLogicTest);
#endif

        default: name=""; break;
    }
}

#endif /* #if !UCONFIG_NO_TRANSLITERATION */
