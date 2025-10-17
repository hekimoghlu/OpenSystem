/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 13, 2023.
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
#ifndef __XPRINTF_DOMAIN_H_
#define __XPRINTF_DOMAIN_H_

#include <sys/cdefs.h>
#include <printf.h>
#include <pthread.h>

#define	PRINTF_TBL_FIRST	'!'
#define	PRINTF_TBL_LAST		'~'
#define PRINTF_TBL_SIZE		(PRINTF_TBL_LAST - PRINTF_TBL_FIRST + 1)

#define printf_tbl_index(x)	((x) - PRINTF_TBL_FIRST)
#define printf_tbl_in_range(x)	((x) >= PRINTF_TBL_FIRST && (x) <= PRINTF_TBL_LAST)

enum {
    PRINTF_DOMAIN_UNUSED = 0,
    PRINTF_DOMAIN_GLIBC_API,
    PRINTF_DOMAIN_FBSD_API,
    PRINTF_DOMAIN_FLAG,
};
#define printf_domain_fbsd_api(d,x)	((d)->type[x] == PRINTF_DOMAIN_FBSD_API)
#define printf_domain_flag(d,x)		((d)->type[x] == PRINTF_DOMAIN_FLAG)
#define printf_domain_glibc_api(d,x)	((d)->type[x] == PRINTF_DOMAIN_GLIBC_API)
#define printf_domain_unused(d,x)	((d)->type[x] == PRINTF_DOMAIN_UNUSED)

struct _printf_tbl {
    printf_arginfo_function *arginfo;
    void *render; /* either typedef printf_function or printf_render */
    void *context;
};
struct _printf_domain {
    pthread_rwlock_t rwlock;
    char type[PRINTF_TBL_SIZE];
    struct _printf_tbl tbl[PRINTF_TBL_SIZE];
};

__BEGIN_DECLS
__END_DECLS

#endif /* __XPRINTF_DOMAIN_H_ */
