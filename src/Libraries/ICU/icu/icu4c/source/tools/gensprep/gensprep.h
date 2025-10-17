/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 23, 2023.
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
*   Copyright (C) 1999-2006, International Business Machines
*   Corporation and others.  All Rights Reserved.
*
*******************************************************************************
*   file name:  gensprep.h
*   encoding:   UTF-8
*   tab size:   8 (not used)
*   indentation:4
*
*   created on: 2003-02-06
*   created by: Ram Viswanadha
*/

#ifndef __GENIDN_H__
#define __GENIDN_H__

#include "unicode/utypes.h"
#include "sprpimpl.h"

/* file definitions */
#define DATA_NAME "sprep"
#define DATA_TYPE "spp"

/*
 * data structure that holds the IDN properties for one or more
 * code point(s) at build time
 */

 
/* global flags */
extern UBool beVerbose, haveCopyright;

/* prototypes */

extern void
setUnicodeVersion(const char *v);

extern void
setUnicodeVersionNC(UVersionInfo version);

extern void
init(void);

#if !UCONFIG_NO_IDNA
extern void
storeMapping(uint32_t codepoint, uint32_t* mapping,int32_t length, UStringPrepType type, UErrorCode* status);
extern void
storeRange(uint32_t start, uint32_t end, UStringPrepType type,UErrorCode* status);
#endif

extern void
generateData(const char *dataDir, const char* bundleName);

extern void
setOptions(int32_t options);

extern void
cleanUpData(void);

/*
extern void
storeIDN(uint32_t code, IDN *idn);

extern void
processData(void);


*/
#endif

/*
 * Hey, Emacs, please set the following:
 *
 * Local Variables:
 * indent-tabs-mode: nil
 * End:
 *
 */
