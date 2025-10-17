/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 28, 2022.
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
#ifdef __APPLE__
/* Satisfy InstallAPI requirements */
#include <sys/types.h>
#include "iconv.h"
#endif

/*
 * Internal prototypes for our back-end functions.
 */
size_t	__bsd___iconv(iconv_t, char **, size_t *, char **,
		size_t *, __uint32_t, size_t *);
void	__bsd___iconv_free_list(char **, size_t);
int	__bsd___iconv_get_list(char ***, size_t *, __iconv_bool);
size_t	__bsd_iconv(iconv_t, char ** __restrict,
		    size_t * __restrict, char ** __restrict,
		    size_t * __restrict);
const char *__bsd_iconv_canonicalize(const char *);
int	__bsd_iconv_close(iconv_t);
iconv_t	__bsd_iconv_open(const char *, const char *);
int	__bsd_iconv_open_into(const char *, const char *, iconv_allocation_t *);
void	__bsd_iconv_set_relocation_prefix(const char *, const char *);
int	__bsd_iconvctl(iconv_t, int, void *);
void	__bsd_iconvlist(int (*) (unsigned int, const char * const *, void *), void *);

