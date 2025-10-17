/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 3, 2024.
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
#ifndef _SYS__PTHREAD_TYPES_H_
#define _SYS__PTHREAD_TYPES_H_

#include <sys/cdefs.h>

// pthread opaque structures
#if defined(__LP64__)
#define __PTHREAD_SIZE__		8176
#define __PTHREAD_ATTR_SIZE__		56
#define __PTHREAD_MUTEXATTR_SIZE__	8
#define __PTHREAD_MUTEX_SIZE__		56
#define __PTHREAD_CONDATTR_SIZE__	8
#define __PTHREAD_COND_SIZE__		40
#define __PTHREAD_ONCE_SIZE__		8
#define __PTHREAD_RWLOCK_SIZE__		192
#define __PTHREAD_RWLOCKATTR_SIZE__	16
#else // !__LP64__
#define __PTHREAD_SIZE__		4088
#define __PTHREAD_ATTR_SIZE__		36
#define __PTHREAD_MUTEXATTR_SIZE__	8
#define __PTHREAD_MUTEX_SIZE__		40
#define __PTHREAD_CONDATTR_SIZE__	4
#define __PTHREAD_COND_SIZE__		24
#define __PTHREAD_ONCE_SIZE__		4
#define __PTHREAD_RWLOCK_SIZE__		124
#define __PTHREAD_RWLOCKATTR_SIZE__	12
#endif // !__LP64__

struct __darwin_pthread_handler_rec {
	void (*__routine)(void *);	// Routine to call
	void *__arg;			// Argument to pass
	struct __darwin_pthread_handler_rec *__next;
};

struct _opaque_pthread_attr_t {
	long __sig;
	char __opaque[__PTHREAD_ATTR_SIZE__];
};

struct _opaque_pthread_cond_t {
	long __sig;
	char __opaque[__PTHREAD_COND_SIZE__];
};

struct _opaque_pthread_condattr_t {
	long __sig;
	char __opaque[__PTHREAD_CONDATTR_SIZE__];
};

struct _opaque_pthread_mutex_t {
	long __sig;
	char __opaque[__PTHREAD_MUTEX_SIZE__];
};

struct _opaque_pthread_mutexattr_t {
	long __sig;
	char __opaque[__PTHREAD_MUTEXATTR_SIZE__];
};

struct _opaque_pthread_once_t {
	long __sig;
	char __opaque[__PTHREAD_ONCE_SIZE__];
};

struct _opaque_pthread_rwlock_t {
	long __sig;
	char __opaque[__PTHREAD_RWLOCK_SIZE__];
};

struct _opaque_pthread_rwlockattr_t {
	long __sig;
	char __opaque[__PTHREAD_RWLOCKATTR_SIZE__];
};

struct _opaque_pthread_t {
	long __sig;
	struct __darwin_pthread_handler_rec  *__cleanup_stack;
	char __opaque[__PTHREAD_SIZE__];
};

typedef struct _opaque_pthread_attr_t __darwin_pthread_attr_t;
typedef struct _opaque_pthread_cond_t __darwin_pthread_cond_t;
typedef struct _opaque_pthread_condattr_t __darwin_pthread_condattr_t;
typedef unsigned long __darwin_pthread_key_t;
typedef struct _opaque_pthread_mutex_t __darwin_pthread_mutex_t;
typedef struct _opaque_pthread_mutexattr_t __darwin_pthread_mutexattr_t;
typedef struct _opaque_pthread_once_t __darwin_pthread_once_t;
typedef struct _opaque_pthread_rwlock_t __darwin_pthread_rwlock_t;
typedef struct _opaque_pthread_rwlockattr_t __darwin_pthread_rwlockattr_t;
typedef struct _opaque_pthread_t *__darwin_pthread_t;

#endif // _SYS__PTHREAD_TYPES_H_
