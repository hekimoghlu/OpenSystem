/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 28, 2024.
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

  zbz2err.c

  This file contains the "fatal error" callback routine required by the
  "minimal" (silent, non-stdio) setup of the bzip2 compression library.

  The fatal bzip2 error bail-out routine is provided in a separate code
  module, so that it can be easily overridden when the Zip package is
  used as a static link library. One example is the WinDLL static library
  usage for building a monolithic binary of the Windows application "WiZ"
  that supports bzip2 both in compression and decompression operations.

  Contains:  bz_internal_error()      (BZIP2_SUPPORT only)

  Adapted from UnZip ubz2err.c, with all the DLL fine print stripped
  out.

  ---------------------------------------------------------------------------*/


#define __ZBZ2ERR_C     /* identifies this source module */

#include "zip.h"

#ifdef BZIP2_SUPPORT
# ifdef BZIP2_USEBZIP2DIR
#   include "bzip2/bzlib.h"
# else
    /* If IZ_BZIP2 is defined as the location of the bzip2 files then
       assume the location has been added to include path.  For Unix
       this is done by the configure script. */
    /* Also do not need path for bzip2 include if OS includes support
       for bzip2 library. */
#   include "bzlib.h"
# endif

/**********************************/
/*  Function bz_internal_error()  */
/**********************************/

/* Call-back function for the bzip2 decompression code (compiled with
 * BZ_NO_STDIO), required to handle fatal internal bug-type errors of
 * the bzip2 library.
 */
void bz_internal_error(errcode)
    int errcode;
{
    sprintf(errbuf, "fatal error (code %d) in bzip2 library", errcode);
    ziperr(ZE_LOGIC, errbuf);
} /* end function bz_internal_error() */

#endif /* def BZIP2_SUPPORT */
