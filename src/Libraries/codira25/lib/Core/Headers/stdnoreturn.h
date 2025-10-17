/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 9, 2023.
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
#ifndef __STDNORETURN_H
#define __STDNORETURN_H

#if defined(__MVS__) && __has_include_next(<stdnoreturn.h>)
#include_next <stdnoreturn.h>
#else

#define noreturn _Noreturn
#define __noreturn_is_defined 1

#endif /* __MVS__ */

#if (defined(__STDC_VERSION__) && __STDC_VERSION__ > 201710L) &&               \
    !defined(_CLANG_DISABLE_CRT_DEPRECATION_WARNINGS)
/* The noreturn macro is deprecated in C23. We do not mark it as such because
   including the header file in C23 is also deprecated and we do not want to
   issue a confusing diagnostic for code which includes <stdnoreturn.h>
   followed by code that writes [[noreturn]]. The issue with such code is not
   with the attribute, or the use of 'noreturn', but the inclusion of the
   header. */
/* FIXME: We should be issuing a deprecation warning here, but cannot yet due
 * to system headers which include this header file unconditionally.
 */
#endif

#endif /* __STDNORETURN_H */
