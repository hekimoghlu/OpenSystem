/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 12, 2021.
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
* File CG7COLL.H
*
* Modification History:
*        Name                     Description            
*     Madhu Katragadda            Converted to C
*********************************************************************************/
/**
 * G7CollationTest is a third level test class.  This test performs the examples 
 * mentioned on the IBM Java international demos web site.  
 * Sample Rules: & Z < p , P 
 * Effect :  Making P sort after Z.
 *
 * Sample Rules: & c < ch , cH, Ch, CH 
 * Effect : As well as adding sequences of characters that act as a single character (this is
 * known as contraction), you can also add characters that act like a sequence of
 * characters (this is known as expansion).  
 * 
 * Sample Rules: & Question'-'mark ; '?' & Hash'-'mark ; '#' & Ampersand ; '&' 
 * Effect : Expansion and contraction can actually be combined.  
 * 
 * Sample Rules: & aa ; a'-' & ee ; e'-' & ii ; i'-' & oo ; o'-' & uu ; u'-'
 * Effect : sorted sequence as the following,
 * aardvark  
 * a-rdvark  
 * abbot  
 * coop  
 * co-p  
 * cop 
 */

#ifndef _CG7COLLTST
#define _CG7COLLTST

#include "unicode/utypes.h"
#include "cmemory.h"

#if !UCONFIG_NO_COLLATION

#include "cintltst.h"

#define MAX_TOKEN_LEN 16
#define  TESTLOCALES  12 
#define  FIXEDTESTSET 15 
#define  TOTALTESTSET  30 

    /* perform test for G7 locales */
    static void TestG7Locales(void);

    /* perform test with added rules " & Z < p, P" */
    static void TestDemo1(void);

    /* perorm test with added rules "& C < ch , cH, Ch, CH" */
    static void TestDemo2(void);

    /* perform test with added rules  */
    /* "& Question'-'mark ; '?' & Hash'-'mark ; '#' & Ampersand ; '&'" */
    static void TestDemo3(void);

    /* perform test with added rules  */
    /* " & aa ; a'-' & ee ; e'-' & ii ; i'-' & oo ; o'-' & uu ; u'-' " */
    static void TestDemo4(void);

#endif /* #if !UCONFIG_NO_COLLATION */

#endif
