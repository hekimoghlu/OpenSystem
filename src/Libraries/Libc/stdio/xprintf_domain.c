/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 5, 2025.
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
#include <printf.h>
#include <pthread.h>
#include <stdlib.h>
#include <errno.h>
#include "xprintf_domain.h"
#include "xprintf_private.h"
#include "libc_private.h" // for LIBC_ABORT

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpointer-bool-conversion"

/* These are flag characters and can never be used as conversion specifiers */
static const char _printf_tbl_flags[] = "#$'*+,-.0123456789:;L_hjlqtvz";

struct _printf_tbl_defaults_fbsd {
    const char *spec;
    printf_arginfo_function *arginfo;
    printf_render *render;
};
static struct _printf_tbl_defaults_fbsd _printf_tbl_defaults_fbsd[] = {
    {"%",		__printf_arginfo_pct,	__printf_render_pct},
    {"AEFGaefg",	__printf_arginfo_float,	__printf_render_float},
    {"Cc",		__printf_arginfo_chr,	__printf_render_chr},
    {"DOUXdioux",	__printf_arginfo_int,	__printf_render_int},
    {"Ss",		__printf_arginfo_str,	__printf_render_str},
    {"p",		__printf_arginfo_ptr,	__printf_render_ptr},
};
struct _printf_tbl_defaults_glibc {
    const char *spec;
    printf_arginfo_function *arginfo;
    printf_function *render;
};
static struct _printf_tbl_defaults_glibc _printf_tbl_defaults_glibc[] = {
    {"n",		__printf_arginfo_n,	__printf_render_n},
};

static printf_domain_t xprintf_domain_default;
#ifdef XPRINTF_DEBUG
__private_extern__ printf_domain_t xprintf_domain_global = NULL;
#endif

__private_extern__ pthread_once_t __xprintf_domain_once = PTHREAD_ONCE_INIT;

__private_extern__ void
__xprintf_domain_init(void)
{
    xprintf_domain_default = (printf_domain_t)calloc(
#ifdef XPRINTF_DEBUG
						     2,
#else
						     1,
#endif
						     sizeof(*xprintf_domain_default));
    if(xprintf_domain_default == NULL)
	LIBC_ABORT("No memory");

    xprintf_domain_default->rwlock = (pthread_rwlock_t)PTHREAD_RWLOCK_INITIALIZER;
    {
	const char *cp;
	for(cp = _printf_tbl_flags; *cp; cp++)
	    xprintf_domain_default->type[printf_tbl_index(*cp)] = PRINTF_DOMAIN_FLAG;
    }
    {
	struct _printf_tbl_defaults_fbsd *d = _printf_tbl_defaults_fbsd;
	int n = sizeof(_printf_tbl_defaults_fbsd) / sizeof(*_printf_tbl_defaults_fbsd);
	for(; n > 0; d++, n--) {
	    for(const char *cp = d->spec; *cp; cp++) {
		xprintf_domain_default->type[printf_tbl_index(*cp)] = PRINTF_DOMAIN_FBSD_API;
		xprintf_domain_default->tbl[printf_tbl_index(*cp)] = (struct _printf_tbl){d->arginfo, d->render, NULL};
	    }
	}
    }
    {
	struct _printf_tbl_defaults_glibc *d = _printf_tbl_defaults_glibc;
	int n = sizeof(_printf_tbl_defaults_glibc) / sizeof(*_printf_tbl_defaults_glibc);
	for(; n > 0; d++, n--) {
	    for(const char *cp = d->spec; *cp; cp++) {
		xprintf_domain_default->type[printf_tbl_index(*cp)] = PRINTF_DOMAIN_GLIBC_API;
		xprintf_domain_default->tbl[printf_tbl_index(*cp)] = (struct _printf_tbl){d->arginfo, d->render, NULL};
	    }
	}
    }
#ifdef XPRINTF_DEBUG
    xprintf_domain_global = xprintf_domain_default + 1;
    *xprintf_domain_global = *xprintf_domain_default;
#endif
}

printf_domain_t
copy_printf_domain(printf_domain_t src)
{
    printf_domain_t restrict copy;

    if(!src) {
	errno = EINVAL;
	return NULL;
    }
    copy = (printf_domain_t)MALLOC(sizeof(*copy));
    if(!copy) return NULL;
    xprintf_domain_init();
    pthread_rwlock_rdlock(&src->rwlock);
    *copy = *src;
    pthread_rwlock_unlock(&src->rwlock);
    copy->rwlock = (pthread_rwlock_t)PTHREAD_RWLOCK_INITIALIZER;
    return copy;
}

void
free_printf_domain(printf_domain_t d)
{
    if(!d) return;
    pthread_rwlock_destroy(&d->rwlock);
    free(d);
}

printf_domain_t
new_printf_domain(void)
{
    printf_domain_t restrict d;

    xprintf_domain_init();

    d = (printf_domain_t)MALLOC(sizeof(*d));
    if(!d) return NULL;
    *d = *xprintf_domain_default;
    return d;
}

int
register_printf_domain_function(printf_domain_t d, int spec, printf_function *render, printf_arginfo_function *arginfo, void *context)
{
    xprintf_domain_init();

    if(!d || !printf_tbl_in_range(spec)) {
	errno = EINVAL;
	return -1;
    }
    xprintf_domain_init();

    switch(d->type[printf_tbl_index(spec)]) {
    case PRINTF_DOMAIN_FLAG:
	errno = EINVAL;
	return -1;
    default:
	pthread_rwlock_wrlock(&d->rwlock);
	if(!render || !arginfo) {
	    d->type[printf_tbl_index(spec)] = PRINTF_DOMAIN_UNUSED;
	} else {
	    d->type[printf_tbl_index(spec)] = PRINTF_DOMAIN_GLIBC_API;
	    d->tbl[printf_tbl_index(spec)] = (struct _printf_tbl){arginfo, render, context};
	}
	pthread_rwlock_unlock(&d->rwlock);
    }

    return 0;
}

__private_extern__ int
register_printf_domain_render(printf_domain_t d, int spec, printf_render *render, printf_arginfo_function *arginfo)
{
    xprintf_domain_init();

    if(!d || !printf_tbl_in_range(spec)) {
	errno = EINVAL;
	return -1;
    }
    xprintf_domain_init();

    switch(d->type[printf_tbl_index(spec)]) {
    case PRINTF_DOMAIN_FLAG:
	errno = EINVAL;
	return -1;
    default:
	pthread_rwlock_wrlock(&d->rwlock);
	if(!render || !arginfo) {
	    d->type[printf_tbl_index(spec)] = PRINTF_DOMAIN_UNUSED;
	} else {
	    d->type[printf_tbl_index(spec)] = PRINTF_DOMAIN_FBSD_API;
	    d->tbl[printf_tbl_index(spec)] = (struct _printf_tbl){arginfo, render, NULL};
	}
	pthread_rwlock_unlock(&d->rwlock);
    }

    return 0;
}

int
register_printf_domain_render_std(printf_domain_t d, const char *specs)
{
    int ret = 0;

    for (; *specs != '\0'; specs++) {
	switch (*specs) {
	case 'H':
	    ret = register_printf_domain_render(d, *specs,
		__printf_render_hexdump,
		__printf_arginfo_hexdump);
	    break;
	case 'M':
	    ret = register_printf_domain_render(d, *specs,
		__printf_render_errno,
		__printf_arginfo_errno);
	    break;
	case 'Q':
	    ret = register_printf_domain_render(d, *specs,
		__printf_render_quote,
		__printf_arginfo_quote);
	    break;
	case 'T':
	    ret = register_printf_domain_render(d, *specs,
		__printf_render_time,
		__printf_arginfo_time);
	    break;
	case 'V':
	    ret = register_printf_domain_render(d, *specs,
		__printf_render_vis,
		__printf_arginfo_vis);
	    break;
	default:
	    errno = EINVAL;
	    return (-1);
	}
	if(ret < 0) return ret;
    }
    return (0);
}
#pragma clang diagnostic pop

