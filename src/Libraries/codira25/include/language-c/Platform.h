/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 7, 2021.
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
#ifndef LANGUAGE_CORE_C_PLATFORM_H
#define LANGUAGE_CORE_C_PLATFORM_H

#include "language/Core-c/ExternC.h"

LANGUAGE_CORE_C_EXTERN_C_BEGIN

/* Windows DLL import/export. */
#ifndef CINDEX_NO_EXPORTS
  #define CINDEX_EXPORTS
#endif
#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef CINDEX_EXPORTS
    #ifdef _CINDEX_LIB_
      #define CINDEX_LINKAGE __declspec(dllexport)
    #else
      #define CINDEX_LINKAGE __declspec(dllimport)
    #endif
  #endif
#elif defined(CINDEX_EXPORTS) && defined(__GNUC__)
  #define CINDEX_LINKAGE __attribute__((visibility("default")))
#endif

#ifndef CINDEX_LINKAGE
  #define CINDEX_LINKAGE
#endif

#ifdef __GNUC__
  #define CINDEX_DEPRECATED __attribute__((deprecated))
#else
  #ifdef _MSC_VER
    #define CINDEX_DEPRECATED __declspec(deprecated)
  #else
    #define CINDEX_DEPRECATED
  #endif
#endif

LANGUAGE_CORE_C_EXTERN_C_END

#endif
