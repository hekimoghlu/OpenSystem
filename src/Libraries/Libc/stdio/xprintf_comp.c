/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 4, 2022.
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
#include <xlocale_private.h>
#include <printf.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "xprintf_domain.h"
#include "xprintf_private.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpointer-bool-conversion"

void
free_printf_comp(printf_comp_t pc)
{
    if(!pc) return;
    xlocale_release(pc->loc);
#ifdef XPRINTF_PERF
    arrayfree(pc->pa);
    free(pc->pa);
    arrayfree(pc->aa);
    free(pc->aa);
    arrayfree(pc->ua);
    free(pc->ua);
#else /* !XPRINTF_PERF */
    free(pc->pi);
    free(pc->argt);
    free(pc->args);
#endif /* !XPRINTF_PERF */
    pthread_mutex_destroy(&pc->mutex);
    free(pc);
}

printf_comp_t
new_printf_comp(printf_domain_t restrict domain, locale_t loc, const char * restrict fmt)
{
    int ret, saverrno;
    printf_comp_t restrict pc;

    if(!domain) {
	errno = EINVAL;
	return NULL;
    }
    pc = MALLOC(sizeof(*pc) + strlen(fmt) + 1);
    if(!pc) return NULL;
    bzero(pc, sizeof(*pc));
    pc->mutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
    pc->fmt = (const char *)(pc + 1);
    strcpy((char *)pc->fmt, fmt);
    DEFAULT_CURRENT_LOCALE(loc);
    xlocale_retain(loc);
    pc->loc = loc;
    xprintf_domain_init();
    pthread_rwlock_rdlock(&domain->rwlock);
    ret = __printf_comp(pc, domain);
    saverrno = errno;
    pthread_rwlock_unlock(&domain->rwlock);
    if(ret < 0) {
	xlocale_release(loc);
	pthread_mutex_destroy(&pc->mutex);
	free(pc);
	errno = saverrno;
	return NULL;
    }
    return pc;
}
#pragma clang diagnostic pop

