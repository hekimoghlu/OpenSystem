/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 28, 2021.
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
#ifndef __DCETHREAD_UTIL_H__
#define __DCETHREAD_UTIL_H__

#include <errno.h>

#include "dcethread-exception.h"

int dcethread__set_errno(int err);

#define DCETHREAD_BEGIN_SYSCALL \
    do                                                                  \
    {                                                                   \
        if (dcethread__begin_block (dcethread__self(), NULL, NULL, NULL, NULL)) \
        {                                                               \
            dcethread__dispatchinterrupt(dcethread__self());            \
            return dcethread__set_errno(EINTR);                         \
        }                                                               \
    } while (0)

#define DCETHREAD_END_SYSCALL                                           \
    do                                                                  \
    {                                                                   \
        if (dcethread__end_block (dcethread__self(), NULL, NULL))       \
        {                                                               \
            dcethread__dispatchinterrupt(dcethread__self());            \
            return dcethread__set_errno(EINTR);                         \
        }                                                               \
    } while (0)

#define DCETHREAD_SYSCALL(type, expr)      \
    do					   \
    {					   \
        type ret;                          \
        int err;                           \
	DCETHREAD_BEGIN_SYSCALL;	   \
    	ret = (expr);			   \
	err = errno;			   \
	DCETHREAD_END_SYSCALL;		   \
        errno = err;			   \
        return ret;			   \
    } while(0);

#define DCETHREAD_WRAP_THROW_TYPE(type, expr)				\
    do									\
    {									\
        type ret = (expr);						\
        if (ret < 0)							\
	    dcethread__exc_raise(dcethread__exc_from_errno(errno), __FILE__, __LINE__); \
	return ret;							\
    } while (0);							\

#define DCETHREAD_WRAP_THROW(expr) DCETHREAD_WRAP_THROW_TYPE(int, expr)

#endif
