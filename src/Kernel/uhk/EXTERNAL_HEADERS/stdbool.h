/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 16, 2022.
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
#if XNU_KERNEL_PRIVATE

#include_next <stdbool.h>
/* __STDBOOL_H guard defined */

#elif (defined(__has_include) && __has_include(<__xnu_libcxx_sentinel.h>))

#if !__has_include_next(<stdbool.h>)
#error Do not build with -nostdinc (use GCC_USE_STANDARD_INCLUDE_SEARCHING=NO)
#else
#include_next <stdbool.h>
/* __STDBOOL_H guard defined */
#endif /* __has_include_next */

#else /* XNU_KERNEL_PRIVATE */

#ifndef _STDBOOL_H_
#define _STDBOOL_H_

#define __bool_true_false_are_defined   1

#ifndef __cplusplus

#define false   0
#define true    1

#define bool    _Bool
#if __STDC_VERSION__ < 199901L && __GNUC__ < 3
typedef int     _Bool;
#endif

#endif /* !__cplusplus */

#endif /* !_STDBOOL_H_ */

#endif /* XNU_KERNEL_PRIVATE */
