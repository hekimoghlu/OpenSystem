/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 11, 2024.
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

#ifndef BZ2_LIB_NAME
#    ifdef __WIN32__
#    define BZ2_LIB_NAME "libbz2.dll"
#    endif /* __WIN32__ */
#    ifdef __APPLE__
#    define BZ2_LIB_NAME "libbz2.dylib"
#    endif /* __APPLE__ */
#    ifndef BZ2_LIB_NAME
#    define BZ2_LIB_NAME "libbz2.so"
#    endif /* BZ2_LIB_NAME */
#endif /* BZ2_LIB_NAME */


static char* symbols [] = {
  "BZ2_bzCompress",
  "BZ2_bzCompressEnd",
  "BZ2_bzCompressInit",
  "BZ2_bzDecompress",
  "BZ2_bzDecompressEnd",
  "BZ2_bzDecompressInit",
  (char *) NULL
};


/*
 * Global variable containing the vectors into the 'bz2'-library.
 */

#ifdef BZLIB_STATIC_BUILD
bzFunctions bz = {
  0,
  bzCompress,
  bzCompressEnd,
  bzCompressInit,
  bzDecompress,
  bzDecompressEnd,
  bzDecompressInit,
};
#else
bzFunctions bz = {0}; /* THREADING: serialize initialization */
#endif


int
TrfLoadBZ2lib (interp)
    Tcl_Interp* interp;
{
#ifndef BZLIB_STATIC_BUILD
  int res;

  TrfLock; /* THREADING: serialize initialization */

  res = Trf_LoadLibrary (interp, BZ2_LIB_NAME, (VOID**) &bz, symbols, 6);
  TrfUnlock;

  return res;
#else
  return TCL_OK;
#endif
}
