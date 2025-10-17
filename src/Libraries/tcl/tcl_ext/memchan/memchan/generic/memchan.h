/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 11, 2023.
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
#ifndef _MEMCHAN_H_INCLUDE 
#define _MEMCHAN_H_INCLUDE

#include <tcl.h>

/*
 * Windows needs to know which symbols to export.  Unix does not.
 * BUILD_Memchan should be undefined for Unix.
 */

#undef TCL_STORAGE_CLASS
#ifdef BUILD_Memchan
#define TCL_STORAGE_CLASS DLLEXPORT
#else
#ifdef USE_MEMCHAN_STUBS
#define TCL_STORAGE_CLASS
#else
#define TCL_STORAGE_CLASS DLLIMPORT
#endif /* USE_MEMCHAN_STUBS */
#endif /* BUILD_Memchan */


#ifdef __cplusplus
extern "C" {
#endif

#include "memchanDecls.h"

#ifdef USE_MEMCHAN_STUBS
EXTERN CONST char * 
Memchan_InitStubs(Tcl_Interp *interp, CONST char *version, int exact);
#endif

#ifdef __cplusplus
}
#endif /* C++ */

#undef TCL_STORAGE_CLASS
#define TCL_STORAGE_CLASS DLLIMPORT

#endif /* _MEMCHAN_H_INCLUDE */
