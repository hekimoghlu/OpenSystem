/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 21, 2025.
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
 * Copyright (c) 1997-2001, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/
/********************************************************************************
*
* File CNORMTST.H
*
* Modification History:
*        Name                     Description            
*     Madhu Katragadda            Converted to C
*     synwee                      added test for quick check
*     synwee                      added test for checkFCD
*********************************************************************************
*/
#ifndef _NORMTST
#define _NORMTST
/**
 *  tests for u_normalization
 */

#include "cintltst.h"
    
    void TestDecomp(void);
    void TestCompatDecomp(void);
    void TestCanonDecompCompose(void);
    void TestCompatDecompCompose(void);
    void TestNull(void);
    void TestQuickCheck(void);
    void TestCheckFCD(void);

    /*internal functions*/
    
/*    static void assertEqual(const UChar* result,  const UChar* expected, int32_t index);
*/
    static void assertEqual(const UChar* result,  const char* expected, int32_t index);


    


#endif
