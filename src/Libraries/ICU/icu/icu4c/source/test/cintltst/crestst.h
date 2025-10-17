/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 4, 2023.
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
 * Copyright (c) 1997-2004, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/
/********************************************************************************
*
* File CRESTST.H
*
* Modification History:
*        Name                     Description            
*     Madhu Katragadda            Converted to C
*********************************************************************************
*/
#ifndef _CRESTST
#define _CRESTST
/* C API TEST FOR RESOURCEBUNDLE */
#include "cintltst.h"




    void addTestResourceBundleTest(TestNode**);

    /**
     * Perform several extensive tests using the subtest routine testTag
     */
    void TestResourceBundles(void);
    /** 
     * Test construction of ResourceBundle accessing a custom test resource-file
     */
    void TestConstruction1(void);

    void TestConstruction2(void);

    void TestAliasConflict(void);

    static void TestGetSize(void);

    static void TestGetLocaleByType(void);

    /**
     * extensive subtests called by TestResourceBundles
     **/

    UBool testTag(const char* frag, UBool in_Root, UBool in_te, UBool in_te_IN);

    void record_pass(void);
    void record_fail(void);


    int32_t pass;
    int32_t fail;

#endif
