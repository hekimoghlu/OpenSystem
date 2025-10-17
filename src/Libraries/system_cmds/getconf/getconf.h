/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 5, 2022.
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
#ifdef STABLE
typedef long long intmax_t;
#define	PRIdMAX	"lld"
#else
#include <inttypes.h>
#endif

#ifdef __APPLE__
#define  nitems(x)   (sizeof((x)) / sizeof((x)[0]))
#endif

int	find_confstr(const char *name, int *key);
int	find_unsigned_limit(const char *name, uintmax_t *value);
int	find_limit(const char *name, intmax_t *value);
int	find_pathconf(const char *name, int *key);
int	find_progenv(const char *name, const char **alt_path);
int	find_sysconf(const char *name, int *key);
void	foreach_confstr(void (*func)(const char *, int));
void	foreach_pathconf(void (*func)(const char *, int, const char *),
	    const char *path);
void	foreach_sysconf(void (*func)(const char *, int));
