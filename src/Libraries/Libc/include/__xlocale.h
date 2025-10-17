/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 23, 2023.
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
#ifndef __XLOCALE_H_
#define __XLOCALE_H_

#include <sys/cdefs.h>
#include <_bounds.h>
#include <_mb_cur_max.h>
#include <_types/_locale_t.h>

_LIBC_SINGLE_BY_DEFAULT()

__BEGIN_DECLS
int		___mb_cur_max_l(locale_t);
__END_DECLS

#undef MB_CUR_MAX_L
#define MB_CUR_MAX_L(x)			(___mb_cur_max_l(x))

#endif /* __XLOCALE_H_ */
