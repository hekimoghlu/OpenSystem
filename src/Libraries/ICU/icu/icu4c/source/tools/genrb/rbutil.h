/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 8, 2022.
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
*   Copyright (C) 1998-2016, International Business Machines
*   Corporation and others.  All Rights Reserved.
*
*******************************************************************************
*
* File rbutil.h
*
* Modification History:
*
*   Date        Name        Description
*   06/10/99    stephen     Creation.
*******************************************************************************
*/

#ifndef UTIL_H
#define UTIL_H 1

#include "unicode/utypes.h"

U_CDECL_BEGIN

void get_dirname(char *dirname, const char *filename);
void get_basename(char *basename, const char *filename);
int32_t itostr(char * buffer, int32_t i, uint32_t radix, int32_t pad);

U_CDECL_END

#endif /* ! UTIL_H */
