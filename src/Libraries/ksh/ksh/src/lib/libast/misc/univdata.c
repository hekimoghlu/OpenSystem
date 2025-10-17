/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 29, 2022.
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
 * Glenn Fowler
 * AT&T Bell Laboratories
 *
 * universe common data
 */

#include "univlib.h"

#ifndef UNIV_MAX

char		univ_env[] = "__UNIVERSE__";

#else

#ifndef NUMUNIV

#if !_lib_universe
#undef	U_GET
#endif

#ifdef	U_GET
char*		univ_name[] = { "ucb", "att" };
#else
char*		univ_name[] = { "att", "ucb" };
#endif

int		univ_max = sizeof(univ_name) / sizeof(univ_name[0]);

#endif

char		univ_cond[] = "$(UNIVERSE)";

int		univ_size = sizeof(univ_cond) - 1;

#endif
