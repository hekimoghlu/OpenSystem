/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 26, 2024.
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
*******************************************************************************
*
*   Copyright (C) 2005-2016, International Business Machines
*   Corporation and others.  All Rights Reserved.
*
*******************************************************************************
*   file name:  pkg_imp.h
*   encoding:   UTF-8
*   tab size:   8 (not used)
*   indentation:4
*
*   created on: 2005sep18
*   created by: Markus W. Scherer
*
*   Implementation definitions for data package functions in toolutil.
*/

#ifndef __PKG_IMP_H__
#define __PKG_IMP_H__

#include "unicode/utypes.h"
#include "unicode/udata.h"

/*
 * Read an ICU data item with any platform type,
 * return the pointer to the UDataInfo in its header,
 * and set the lengths of the UDataInfo and of the whole header.
 * All data remains in its platform type.
 */
U_CFUNC const UDataInfo *
getDataInfo(const uint8_t *data, int32_t length,
            int32_t &infoLength, int32_t &headerLength,
            UErrorCode *pErrorCode);

#endif
