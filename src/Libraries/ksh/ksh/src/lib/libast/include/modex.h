/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 28, 2023.
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
#pragma prototyped
/*
 * AT&T Research
 *
 * external mode_t representation support
 */

#ifndef _MODEX_H
#define _MODEX_H

#include <ast_fs.h>
#include <modecanon.h>

#if _BLD_ast && defined(__EXPORT__)
#define extern		__EXPORT__
#endif

extern int		modei(int);
extern int		modex(int);

#undef	extern

#if _S_IDPERM
#define modei(m)	((m)&X_IPERM)
#if _S_IDTYPE
#define modex(m)	(m)
#endif
#endif

#endif
