/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 10, 2022.
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
#ifndef _UTILS_H
#define _UTILS_H


#include <syslog.h>
#include <sys/types.h>
#include <sys/uio.h>
#include "saslauthd.h"


/* log prioities */
#define L_ERR			LOG_ERR
#define L_INFO			LOG_INFO
#define L_DEBUG			LOG_DEBUG


/* some magic to grab function names */
#ifdef HAVE_FUNC
# define L_FUNC __func__
# define HAVE_L_FUNC 1
#elif defined(HAVE_PRETTY_FUNCTION)
# define L_FUNC __PRETTY_FUNCTION__
# define HAVE_L_FUNC 1
#elif defined(HAVE_FUNCTION)
# define L_FUNC __FUNCTION__
# define HAVE_L_FUNC 1
#else
# define L_FUNC ""
# undef HAVE_L_FUNC
#endif

#ifdef HAVE_L_FUNC
# define L_STDERR_FORMAT        "saslauthd[%d] :%-16s: %s\n"
#else
# define L_STDERR_FORMAT        "saslauthd[%d] :%s%s\n"
#endif 


/* utils.c */
extern void	logger(int, const char *, const char *, ...);
extern ssize_t	tx_rec(int filefd, void *, size_t);
extern ssize_t	rx_rec(int , void *, size_t);
extern int	retry_writev(int, struct iovec *, int);


#endif  /* _UTILS_H */
