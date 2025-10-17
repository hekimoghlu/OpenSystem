/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 28, 2025.
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
 * Copyright (c) 1997-2012, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/
/********************************************************************************
*
* File CCONVTST.C
*
* Modification History:
*        Name                     Description            
*     Madhu Katragadda               Creation
*********************************************************************************
*/
#include "cintltst.h"

void addTestConvert(TestNode**);
#include "nucnvtst.h"
void addTestConvertErrorCallBack(TestNode** root);
void addTestEuroRegression(TestNode** root);
void addTestConverterFallBack(TestNode** root);
void addExtraTests(TestNode** root);

/* bocu1tst.c */
U_CFUNC void
addBOCU1Tests(TestNode** root);

void addConvert(TestNode** root);

void addConvert(TestNode** root)
{
    addTestConvert(root);
    addTestNewConvert(root);
    addBOCU1Tests(root);
    addTestConvertErrorCallBack(root);
    addTestEuroRegression(root);
#if !UCONFIG_NO_LEGACY_CONVERSION
    addTestConverterFallBack(root);
#endif
    addExtraTests(root);
}
