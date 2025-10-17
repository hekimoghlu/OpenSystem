/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 10, 2023.
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
/*
 * Thread-aware utmpx SPI
 */
#ifndef _UTMPX_THREAD_H_
#define _UTMPX_THREAD_H_

#include <utmpx.h>

struct _utmpx; /* forward reference */
typedef struct _utmpx *utmpx_t;

__BEGIN_DECLS
int		 _closeutx(utmpx_t);
void    	 _endutxent(utmpx_t);
struct utmpx	*_getutxent(utmpx_t);
struct utmpx	*_getutxid(utmpx_t, const struct utmpx *);
struct utmpx	*_getutxline(utmpx_t, const struct utmpx *);
utmpx_t		 _openutx(const char *);
struct utmpx	*_pututxline(utmpx_t, const struct utmpx *);
void    	 _setutxent(utmpx_t);
int		 _utmpxname(utmpx_t, const char *);
__END_DECLS

#endif /* !_UTMPX_THREAD_H_ */
