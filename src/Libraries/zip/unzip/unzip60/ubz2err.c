/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 31, 2021.
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
/*---------------------------------------------------------------------------

  ubz2err.c

  This file contains the "fatal error" callback routine required by the
  "minimal" (silent, non-stdio) setup of the bzip2 compression library.

  The fatal bzip2 error bail-out routine is provided in a separate code
  module, so that it can be easily overridden when the UnZip package is
  used as a static link library. One example is the WinDLL static library
  usage for building a monolythic binary of the Windows application "WiZ"
  that supports bzip2 both in compression and decompression operations.

  Contains:  bz_internal_error()      (USE_BZIP2 only)

  ---------------------------------------------------------------------------*/


#define __UBZ2ERR_C     /* identifies this source module */
#define UNZIP_INTERNAL
#include "unzip.h"
#ifdef WINDLL
#  ifdef POCKET_UNZIP
#    include "wince/intrface.h"
#  else
#    include "windll/windll.h"
#  endif
#endif

#ifdef USE_BZIP2

/**********************************/
/*  Function bz_internal_error()  */
/**********************************/

/* Call-back function for the bzip2 decompression code (compiled with
 * BZ_NO_STDIO), required to handle fatal internal bug-type errors of
 * the bzip2 library.
 */
void bz_internal_error(bzerrcode)
    int bzerrcode;
{
    GETGLOBALS();

    Info(slide, 0x421, ((char *)slide,
      "error: internal fatal libbzip2 error number %d\n", bzerrcode));
#ifdef WINDLL
    longjmp(dll_error_return, 1);
#else
    DESTROYGLOBALS();
    EXIT(PK_BADERR);
#endif
} /* end function bz_internal_error() */

#endif /* USE_BZIP2 */
