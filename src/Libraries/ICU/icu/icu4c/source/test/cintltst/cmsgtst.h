/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 23, 2023.
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
 * Copyright (c) 1997-2010, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************
 *
 * File CMSGTST.H
 *
 * Modification History:
 *        Name                     Description            
 *     Madhu Katragadda              Creation
 ********************************************************************/
/* C API TEST FOR MESSAGE FORMAT */
#ifndef _CMSGFRMTST
#define _CMSGFRMTST

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "cintltst.h"


/* The function used to test the Message format API*/

    /**
     * Test u_formatMessage() with various test patterns
     **/
    static void MessageFormatTest(void);
    /**
     * Test u_formatMessage() with sample test Patterns 
     **/
    static void TestSampleMessageFormat(void);
    /**
     * Test format and parse sequence and roundtrip
     **/
    static void TestSampleFormatAndParse(void);
    /**
     * Test u_formatMessage() with choice option
     **/
    static void TestMsgFormatChoice(void);
    /**
     * Test u_formatMessage() with Select option
     **/
    static void TestMsgFormatSelect(void);
    /**
     * Test u_parseMessage() with various test patterns()
     **/
    static void TestParseMessage(void);
    /**
     * function used to set up various patterns used for testing u_formatMessage()
     **/
    static void InitStrings( void );

    /**
     * Regression test for ICU4C Jitterbug 904
     */
    static void TestJ904(void);

#endif /* #if !UCONFIG_NO_FORMATTING */

#endif
