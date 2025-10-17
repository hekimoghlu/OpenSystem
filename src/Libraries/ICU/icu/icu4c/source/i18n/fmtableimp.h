/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 18, 2024.
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
* Copyright (C) 2010-2014, International Business Machines Corporation and    *
* others. All Rights Reserved.                                                *
*******************************************************************************
*/

#ifndef FMTABLEIMP_H
#define FMTABLEIMP_H

#include "number_decimalquantity.h"

#if !UCONFIG_NO_FORMATTING

U_NAMESPACE_BEGIN

/** 
 * Maximum int64_t value that can be stored in a double without chancing losing precision.
 *   IEEE doubles have 53 bits of mantissa, 10 bits exponent, 1 bit sign.
 *   IBM Mainframes have 56 bits of mantissa, 7 bits of base 16 exponent, 1 bit sign.
 * Define this constant to the smallest value from those for supported platforms.
 * @internal
 */
static const int64_t MAX_INT64_IN_DOUBLE = 0x001FFFFFFFFFFFFFLL;

U_NAMESPACE_END

#endif // #if !UCONFIG_NO_FORMATTING
#endif
