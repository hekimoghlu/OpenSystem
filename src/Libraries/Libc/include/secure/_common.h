/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 4, 2024.
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
#ifndef _SECURE__COMMON_H_
#define _SECURE__COMMON_H_

#include <_types.h>

#undef _USE_FORTIFY_LEVEL
#if defined(_FORTIFY_SOURCE) && _FORTIFY_SOURCE > 0
#  if _FORTIFY_SOURCE > 1
#    define _USE_FORTIFY_LEVEL 2
#  else
#    define _USE_FORTIFY_LEVEL 1
#  endif
#else
#  define _USE_FORTIFY_LEVEL 0
#endif

#define __darwin_obsz0(object) __builtin_object_size (object, 0)
#define __darwin_obsz(object) __builtin_object_size (object, _USE_FORTIFY_LEVEL > 1 ? 1 : 0)

#endif
