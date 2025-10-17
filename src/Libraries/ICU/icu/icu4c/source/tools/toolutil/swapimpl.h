/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 18, 2022.
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
*   Copyright (C) 2005, International Business Machines
*   Corporation and others.  All Rights Reserved.
*
*******************************************************************************
*   file name:  swapimpl.h
*   encoding:   UTF-8
*   tab size:   8 (not used)
*   indentation:4
*
*   created on: 2005jul29
*   created by: Markus W. Scherer
*
*   Declarations for data file swapping functions not declared in internal
*   library headers.
*/

#ifndef __SWAPIMPL_H__
#define __SWAPIMPL_H__

#include "unicode/utypes.h"
#include "udataswp.h"

/**
 * Identifies and then transforms the ICU data piece in-place, or determines
 * its length. See UDataSwapFn.
 * This function handles single data pieces (but not .dat data packages)
 * and internally dispatches to per-type swap functions.
 * Sets a U_UNSUPPORTED_ERROR if the data format is not recognized.
 *
 * @see UDataSwapFn
 * @see udata_openSwapper
 * @see udata_openSwapperForInputData
 * @internal ICU 2.8
 */
U_CAPI int32_t U_EXPORT2
udata_swap(const UDataSwapper *ds,
           const void *inData, int32_t length, void *outData,
           UErrorCode *pErrorCode);

#endif
