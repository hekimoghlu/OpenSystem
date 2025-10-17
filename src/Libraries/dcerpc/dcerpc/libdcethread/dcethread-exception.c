/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 9, 2025.
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
#include <config.h>
#include <string.h>
#include <setjmp.h>
#include <errno.h>
#include <stdio.h>
#ifdef HAVE_EXECINFO_H
#    include <execinfo.h>
#endif
#include <assert.h>

#include "dcethread-exception.h"
#include "dcethread-debug.h"
#include "dcethread-private.h"

static pthread_key_t frame_key;
static void (*uncaught_handler) (dcethread_exc* exc, const char* file, unsigned int line, void* data);
static void* uncaught_handler_data;

dcethread_exc dcethread_uninitexc_e;
dcethread_exc dcethread_exquota_e;
dcethread_exc dcethread_insfmem_e;
dcethread_exc dcethread_nopriv_e;
dcethread_exc dcethread_illaddr_e;
dcethread_exc dcethread_illinstr_e;
dcethread_exc dcethread_resaddr_e;
dcethread_exc dcethread_privinst_e;
dcethread_exc dcethread_resoper_e;
dcethread_exc dcethread_aritherr_e;
dcethread_exc dcethread_intovf_e;
dcethread_exc dcethread_intdiv_e;
dcethread_exc dcethread_fltovf_e;
dcethread_exc dcethread_fltdiv_e;
dcethread_exc dcethread_fltund_e;
dcethread_exc dcethread_decovf_e;
dcethread_exc dcethread_subrng_e;
dcethread_exc dcethread_excpu_e;
dcethread_exc dcethread_exfilsiz_e;
dcethread_exc dcethread_SIGTRAP_e;
dcethread_exc dcethread_SIGIOT_e;
dcethread_exc dcethread_SIGEMT_e;
dcethread_exc dcethread_SIGSYS_e;
dcethread_exc dcethread_SIGPIPE_e;
dcethread_exc dcethread_unksyncsig_e;
dcethread_exc dcethread_interrupt_e;
dcethread_exc dcethread_badparam_e;           /* Bad parameter */
dcethread_exc dcethread_existence_e;          /* Object does not exist */
dcethread_exc dcethread_in_use_e;             /* Object is in use */
dcethread_exc dcethread_use_error_e;          /* Object inappropriate for operation */
dcethread_exc dcethread_nostackmem_e;         /* No memory to allocate stack */
dcethread_exc dcethread_exit_thread_e;        /* Used to terminate a thread */

static void
default_uncaught_handler(dcethread_exc* exc, const char* file,
	unsigned int line, void* data ATTRIBUTE_UNUSED)
{
    if (!dcethread__exc_matches(exc, &dcethread_interrupt_e) &&
        !dcethread__exc_matches(exc, &dcethread_exit_thread_e))
    {
        const char* name = dcethread__exc_getname(exc);
        if (name)
        {
            fprintf(stderr, "%s:%i: uncaught exception %s in thread %p\n", file, line, name, dcethread__self());
        }
        else
        {
            fprintf(stderr, "%s:%i: uncaught exception %p (%i) in thread %p\n",
                    file, line, exc, dcethread__exc_getstatus(exc), dcethread__self());
        }

#ifdef HAVE_BACKTRACE_SYMBOLS_FD
        void* buffer[256];
        int size;

        size = backtrace(buffer, 256);

        fprintf(stderr, "Backtrace:\n");
        backtrace_symbols_fd(buffer, size, fileno(stderr));
#endif
        abort();
    }

    pthread_exit(0);
}

void
dcethread__init_exceptions(void)
{
    pthread_key_create(&frame_key, NULL);
    uncaught_handler = default_uncaught_handler;

    DCETHREAD_EXC_INIT(dcethread_uninitexc_e);
    DCETHREAD_EXC_INIT(dcethread_exquota_e);
    DCETHREAD_EXC_INIT(dcethread_insfmem_e);
    DCETHREAD_EXC_INIT(dcethread_nopriv_e);
    DCETHREAD_EXC_INIT(dcethread_illaddr_e);
    DCETHREAD_EXC_INIT(dcethread_illinstr_e);
    DCETHREAD_EXC_INIT(dcethread_resaddr_e);
    DCETHREAD_EXC_INIT(dcethread_privinst_e);
    DCETHREAD_EXC_INIT(dcethread_resoper_e);
    DCETHREAD_EXC_INIT(dcethread_aritherr_e);
    DCETHREAD_EXC_INIT(dcethread_intovf_e);
    DCETHREAD_EXC_INIT(dcethread_intdiv_e);
    DCETHREAD_EXC_INIT(dcethread_fltovf_e);
    DCETHREAD_EXC_INIT(dcethread_fltdiv_e);
    DCETHREAD_EXC_INIT(dcethread_fltund_e);
    DCETHREAD_EXC_INIT(dcethread_decovf_e);
    DCETHREAD_EXC_INIT(dcethread_subrng_e);
    DCETHREAD_EXC_INIT(dcethread_excpu_e);
    DCETHREAD_EXC_INIT(dcethread_exfilsiz_e);
    DCETHREAD_EXC_INIT(dcethread_SIGTRAP_e);
    DCETHREAD_EXC_INIT(dcethread_SIGIOT_e);
    DCETHREAD_EXC_INIT(dcethread_SIGEMT_e);
    DCETHREAD_EXC_INIT(dcethread_SIGSYS_e);
    DCETHREAD_EXC_INIT(dcethread_SIGPIPE_e);
    DCETHREAD_EXC_INIT(dcethread_unksyncsig_e);
    DCETHREAD_EXC_INIT(dcethread_interrupt_e);
    DCETHREAD_EXC_INIT(dcethread_badparam_e);
    DCETHREAD_EXC_INIT(dcethread_existence_e);
    DCETHREAD_EXC_INIT(dcethread_in_use_e);
    DCETHREAD_EXC_INIT(dcethread_use_error_e);
    DCETHREAD_EXC_INIT(dcethread_nostackmem_e);
    DCETHREAD_EXC_INIT(dcethread_exit_thread_e);
}

void
dcethread__frame_push(dcethread_frame* frame)
{
    dcethread_frame* cur = pthread_getspecific(frame_key);
    void *pframe = (void*)(struct _dcethread_frame*) frame;

    memset(pframe, 0, sizeof(*frame));

    frame->parent = cur;

    pthread_setspecific(frame_key, (void*) frame);
}

void
dcethread__frame_pop(dcethread_frame* frame)
{
    dcethread_frame* cur = pthread_getspecific(frame_key);

    if (cur == frame)
    {
	pthread_setspecific(frame_key, (void*) frame->parent);
    }
    else
    {
	DCETHREAD_ERROR("Attempted to pop exception frame in incorrect order");
    }
}

void
dcethread__exc_init(dcethread_exc* exc, const char* name)
{
    exc->kind = DCETHREAD_EXC_KIND_ADDRESS;
    exc->match.address = exc;
    exc->name = name;
}

void
dcethread__exc_setstatus(dcethread_exc* exc, int value)
{
    exc->kind = DCETHREAD_EXC_KIND_STATUS;
    exc->match.value = value;
}

int
dcethread__exc_getstatus(dcethread_exc* exc)
{
    if (exc->kind == DCETHREAD_EXC_KIND_STATUS)
	return exc->match.value;
    else
	return -1;
}

const char*
dcethread__exc_getname(dcethread_exc* exc)
{
    if (exc->kind == DCETHREAD_EXC_KIND_STATUS)
    {
        return exc->name;
    }
    else
    {
        return ((dcethread_exc*) exc->match.address)->name;
    }
}

int
dcethread__exc_matches(dcethread_exc* exc, dcethread_exc* pattern)
{
    assert (exc != NULL);
    assert (pattern != NULL);

    return (exc->kind == pattern->kind &&
	    (exc->kind == DCETHREAD_EXC_KIND_STATUS ?
	     exc->match.value == pattern->match.value :
	     exc->match.address == pattern->match.address));
}

void
dcethread__exc_raise(dcethread_exc* exc, const char* file, unsigned int line)
{
    dcethread_frame* cur;

    /* Ensure thread system is initialized */
    dcethread__init();

    cur = pthread_getspecific(frame_key);

    if (cur)
    {
        cur->exc = *exc;
        cur->file = file;
        cur->line = line;
        siglongjmp(((struct _dcethread_frame*) cur)->jmpbuf, 1);
    }
    else
    {
        uncaught_handler(exc, file, line, uncaught_handler_data);
        abort();
    }
}

void
dcethread__exc_handle_interrupt(dcethread* thread ATTRIBUTE_UNUSED, void* data)
{
    dcethread__exc_raise((dcethread_exc*) data, NULL, 0);
}

dcethread_exc*
dcethread__exc_from_errno(int err)
{
    switch (err)
    {
    case EINVAL:    return &dcethread_badparam_e;
    case ERANGE:    return &dcethread_badparam_e;
    case EDEADLK:   return &dcethread_in_use_e;
    case EBUSY:     return &dcethread_in_use_e;
    case EAGAIN:    return &dcethread_in_use_e;
    case ENOMEM:    return &dcethread_insfmem_e;
    case EPERM:     return &dcethread_nopriv_e;
    case -1:        return &dcethread_interrupt_e; /* XXX */
    default:        return &dcethread_use_error_e;
    }
}

void
dcethread__exc_set_uncaught_handler(void (*handler) (dcethread_exc*, const char*, unsigned int, void*), void* data)
{
    uncaught_handler = handler;
    uncaught_handler_data = data;
}
