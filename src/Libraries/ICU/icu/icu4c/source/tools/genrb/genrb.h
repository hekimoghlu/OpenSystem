/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 25, 2024.
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
*   Copyright (C) 2002-2015, International Business Machines
*   Corporation and others.  All Rights Reserved.
*
*******************************************************************************
*
* File genrb.h
*/

#ifndef GENRB_H
#define GENRB_H

#include <stdio.h>
#include "unicode/utypes.h"
#include "unicode/putil.h"
#include "cmemory.h"
#include "cstring.h"
#include "filestrm.h"


#include "ucbuf.h"
#include "errmsg.h"
#include "parse.h"
#include "rbutil.h"

#include "toolutil.h"
#include "uoptions.h"

#include "unicode/ucol.h"
#include "unicode/uloc.h"

/* The version of genrb */
#define GENRB_VERSION "56"

U_CDECL_BEGIN

U_CAPI void processFile(
    const char *filename,
    const char* cp,
    const char *inputDir,
    const char *outputDir,
    const char *packageName,
    UBool omitBinaryCollation,
    UErrorCode *status);

U_CDECL_END

#endif
