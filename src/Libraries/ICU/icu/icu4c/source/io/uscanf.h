/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 8, 2023.
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
******************************************************************************
*
*   Copyright (C) 1998-2014, International Business Machines
*   Corporation and others.  All Rights Reserved.
*
******************************************************************************
*
* File uscanf.h
*
* Modification History:
*
*   Date        Name        Description
*   12/02/98    stephen        Creation.
*   03/13/99    stephen     Modified for new C API.
******************************************************************************
*/

#ifndef USCANF_H
#define USCANF_H

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING && !UCONFIG_NO_CONVERSION

#include "unicode/ustdio.h"

U_CFUNC int32_t
u_scanf_parse(UFILE     *f,
            const UChar *patternSpecification,
            va_list     ap);

#endif /* #if !UCONFIG_NO_FORMATTING */

#endif

