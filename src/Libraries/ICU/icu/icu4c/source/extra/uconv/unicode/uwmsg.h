/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 10, 2024.
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
**********************************************************************
* Copyright (C) 2000-2004, International Business Machines Corporation 
* and others.  All Rights Reserved.
**********************************************************************

Get a message out of the default resource bundle, messageformat it,
and print it to stderr
*/

#ifndef _UWMSG
#define _UWMSG

#include <stdio.h>

#include "unicode/ures.h"

/* Set the path to wmsg's bundle.
   Caller owns storage.
*/
U_CFUNC UResourceBundle *u_wmsg_setPath(const char *path, UErrorCode *err);

/* Format a message and print it's output to a given file stream */
U_CFUNC int u_wmsg(FILE *fp, const char *tag, ... );

/* format an error message */
U_CFUNC const UChar* u_wmsg_errorName(UErrorCode err);

#endif
