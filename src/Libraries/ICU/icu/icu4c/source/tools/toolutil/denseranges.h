/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 8, 2022.
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
*   Copyright (C) 2010, International Business Machines
*   Corporation and others.  All Rights Reserved.
*******************************************************************************
*   file name:  denseranges.h
*   encoding:   UTF-8
*   tab size:   8 (not used)
*   indentation:4
*
*   created on: 2010sep25
*   created by: Markus W. Scherer
*
* Helper code for finding a small number of dense ranges.
*/

#ifndef __DENSERANGES_H__
#define __DENSERANGES_H__

#include "unicode/utypes.h"

/**
 * Does it make sense to write 1..capacity ranges?
 * Returns 0 if not, otherwise the number of ranges.
 * @param values Sorted array of signed-integer values.
 * @param length Number of values.
 * @param density Minimum average range density, in 256th. (0x100=100%=perfectly dense.)
 *                Should be 0x80..0x100, must be 1..0x100.
 * @param ranges Output ranges array.
 * @param capacity Maximum number of ranges.
 * @return Minimum number of ranges (at most capacity) that have the desired density,
 *         or 0 if that density cannot be achieved.
 */
U_CAPI int32_t U_EXPORT2
uprv_makeDenseRanges(const int32_t values[], int32_t length,
                     int32_t density,
                     int32_t ranges[][2], int32_t capacity);

#endif  // __DENSERANGES_H__
