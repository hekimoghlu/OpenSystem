/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 11, 2023.
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
/* inflate.h for UnZip -- by Mark Adler
   version c14f, 23 November 1995 */


/* Copyright history:
   - Starting with UnZip 5.41 of 16-April-2000, this source file
     is covered by the Info-Zip LICENSE cited above.
   - Prior versions of this source file, found in UnZip source packages
     up to UnZip 5.40, were put in the public domain.
     The original copyright note by Mark Adler was:
         "You can do whatever you like with this source file,
         though I would prefer that if you modify it and
         redistribute it that you include comments to that effect
         with your name and the date.  Thank you."

   History:
   vers    date          who           what
   ----  ---------  --------------  ------------------------------------
    c14  12 Mar 93  M. Adler        made inflate.c standalone with the
                                    introduction of inflate.h.
    c14d 28 Aug 93  G. Roelofs      replaced flush/FlushOutput with new version
    c14e 29 Sep 93  G. Roelofs      moved everything into unzip.h; added crypt.h
    c14f 23 Nov 95  G. Roelofs      added UNZIP_INTERNAL to accommodate newly
                                    split unzip.h
 */

#define UNZIP_INTERNAL
#include "unzip.h"     /* provides slide[], typedefs and macros */
#ifdef FUNZIP
#  include "crypt.h"   /* provides NEXTBYTE macro for crypt version of funzip */
#endif
