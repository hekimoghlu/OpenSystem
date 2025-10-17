/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 27, 2022.
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

#ifndef __PYX_HAVE__scipy__special___ufuncs
#define __PYX_HAVE__scipy__special___ufuncs


#ifndef __PYX_HAVE_API__scipy__special___ufuncs

#ifndef __PYX_EXTERN_C
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

__PYX_EXTERN_C DL_IMPORT(int) wrap_PyUFunc_getfperr(void);

#endif /* !__PYX_HAVE_API__scipy__special___ufuncs */

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC init_ufuncs(void);
#else
PyMODINIT_FUNC PyInit__ufuncs(void);
#endif

#endif /* !__PYX_HAVE__scipy__special___ufuncs */
