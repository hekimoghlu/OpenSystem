/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 25, 2025.
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
#include "transformInt.h"

#ifdef __WIN32__
#define Z_LIB_NAME "zlib.dll"
#endif

#ifndef Z_LIB_NAME
#define Z_LIB_NAME LIBZ_DEFAULTNAME
#endif


static char* symbols [] = {
  "deflate",
  "deflateEnd",
  "deflateInit2_",
  "deflateReset",
  "inflate",
  "inflateEnd",
  "inflateInit2_",
  "inflateReset",
  "adler32",
  "crc32",
  (char *) NULL
};


/*
 * Global variable containing the vectors into the 'zlib'-library.
 */

#ifdef ZLIB_STATIC_BUILD
zFunctions zf = {
  0,
  deflate,
  deflateEnd,
  deflateInit_,
  deflateReset,
  inflate,
  inflateEnd,
  inflateInit_,
  inflateReset,
  adler32,
  crc32,
};
#else
zFunctions zf = {0}; /* THREADING: serialize initialization */
#endif

int
TrfLoadZlib (interp)
    Tcl_Interp* interp;
{
#ifndef ZLIB_STATIC_BUILD
  int res;

#ifdef HAVE_zlibtcl_PACKAGE
  /* Try to use zlibtcl first. This makes loading much easier.
   */

  if (Zlibtcl_InitStubs(interp, ZLIBTCL_VERSION, 0) != NULL) {
    /*
     * Copy stub information into the table the rest of Trf is using.
     */

    zf.handle         = 0;
    zf.zdeflate       = deflate      ;
    zf.zdeflateEnd    = deflateEnd   ;
    zf.zdeflateInit2_ = deflateInit2_;
    zf.zdeflateReset  = deflateReset ;
    zf.zinflate       = inflate      ;
    zf.zinflateEnd    = inflateEnd   ;
    zf.zinflateInit2_ = inflateInit2_;
    zf.zinflateReset  = inflateReset ;
    zf.zadler32       = adler32      ;
    zf.zcrc32         = crc32        ;
    return TCL_OK;
  }

#endif

  TrfLock; /* THREADING: serialize initialization */
  res = Trf_LoadLibrary (interp, Z_LIB_NAME, (VOID**) &zf, symbols, 10);
  TrfUnlock;

  return res;
#else
  return TCL_OK;
#endif
}
