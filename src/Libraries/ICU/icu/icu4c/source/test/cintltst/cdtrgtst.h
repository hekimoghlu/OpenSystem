/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 5, 2023.
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
 * Copyright (c) 1997-2002,2008, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/
/********************************************************************************
*
* File CDTRGTST.H
*
* Modification History:
*        Name                     Description            
*     Madhu Katragadda            Converted to C
*********************************************************************************
*/
/* REGRESSION TEST FOR DATE FORMAT */
#ifndef _CDTFRRGSTST
#define _CDTFRRGSTST

#include "unicode/utypes.h"
#include "unicode/udat.h"

#if !UCONFIG_NO_FORMATTING

#include "cintltst.h"

    /**
     * DateFormat Regression tests
     **/

    void Test4029195(void); 
    void Test4056591(void); 
    void Test4059917(void);
    void Test4060212(void); 
    void Test4061287(void); 
    void Test4073003(void); 
    void Test4162071(void); 
    void Test714(void);
    void Test_GEec(void);

    /**
     * test subroutine
     **/
    void aux917(UDateFormat *fmt, UChar* str );

    /**
     * test subroutine used by the testing functions
     **/
    UChar* myFormatit(UDateFormat* datdef, UDate d1);

#endif /* #if !UCONFIG_NO_FORMATTING */

#endif
