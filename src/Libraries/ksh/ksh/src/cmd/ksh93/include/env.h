/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 13, 2025.
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
#ifndef  _ENV_H
#define	_ENV_H	1

#ifdef _BLD_env
#    ifdef __EXPORT__
#	define export	__EXPORT__
#    endif
#else
     typedef void *Env_t;
#endif

/* for use with env_open */
#define ENV_STABLE	(-1)

/* for third agument to env_add */
#define ENV_MALLOCED	1
#define ENV_STRDUP	2

extern void	env_close(Env_t*);
extern int	env_add(Env_t*, const char*, int);
extern int	env_delete(Env_t*, const char*);
extern char	**env_get(Env_t*);
extern Env_t	*env_open(char**,int);
extern Env_t	*env_scope(Env_t*,int);

#undef extern

#endif


