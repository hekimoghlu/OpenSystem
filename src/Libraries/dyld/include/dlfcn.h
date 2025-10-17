/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 16, 2023.
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
  Based on the dlcompat work done by:
		Jorge Acereda  <jacereda@users.sourceforge.net> &
		Peter O'Gorman <ogorman@users.sourceforge.net>
*/

#ifndef _DLFCN_H_
#define _DLFCN_H_

#include <sys/cdefs.h>

#if !defined(_POSIX_C_SOURCE) || defined(_DARWIN_C_SOURCE)
#include <stdbool.h>
#include <Availability.h>

#define __DYLDDL_UNAVAILABLE       __API_UNAVAILABLE(driverkit)
#ifndef __APPLE_BLEACH_SDK__
#define __DYLDDL_DLSYM_UNAVAILABLE __SPI_AVAILABLE(driverkit(19.0))
#else
#define __DYLDDL_DLSYM_UNAVAILABLE __API_UNAVAILABLE(driverkit)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Structure filled in by dladdr().
 */
typedef struct dl_info {
        const char      *dli_fname;     /* Pathname of shared object */
        void            *dli_fbase;     /* Base address of shared object */
        const char      *dli_sname;     /* Name of nearest symbol */
        void            *dli_saddr;     /* Address of nearest symbol */
} Dl_info;

extern int dladdr(const void *, Dl_info *);

#ifdef __cplusplus
}
#endif

#else
#define __DYLDDL_UNAVAILABLE
#define __DYLDDL_DLSYM_UNAVAILABLE
#endif /* not POSIX */

#ifdef __cplusplus
extern "C" {
#endif

extern int dlclose(void * __handle) __DYLDDL_UNAVAILABLE;
extern char * dlerror(void) __DYLDDL_UNAVAILABLE;
extern void * dlopen(const char * __path, int __mode) __DYLDDL_UNAVAILABLE;
extern void * dlsym(void * __handle, const char * __symbol) __DYLDDL_DLSYM_UNAVAILABLE;

#if !defined(_POSIX_C_SOURCE) || defined(_DARWIN_C_SOURCE)
extern bool dlopen_preflight(const char* __path) __API_AVAILABLE(macos(10.5), ios(2.0)) __DYLDDL_UNAVAILABLE;
#endif /* not POSIX */


#define RTLD_LAZY	0x1
#define RTLD_NOW	0x2
#define RTLD_LOCAL	0x4
#define RTLD_GLOBAL	0x8

#if !defined(_POSIX_C_SOURCE) || defined(_DARWIN_C_SOURCE)
#define RTLD_NOLOAD	0x10
#define RTLD_NODELETE	0x80
#define RTLD_FIRST	0x100	/* Mac OS X 10.5 and later */

/*
 * Special handle arguments for dlsym().
 */
#define	RTLD_NEXT	((void *) -1)	/* Search subsequent objects. */
#define	RTLD_DEFAULT	((void *) -2)	/* Use default search algorithm. */
#define	RTLD_SELF	((void *) -3)	/* Search this and subsequent objects (Mac OS X 10.5 and later) */
#define	RTLD_MAIN_ONLY	((void *) -5)	/* Search main executable only (Mac OS X 10.5 and later) */
#endif /* not POSIX */

#ifdef __cplusplus
}
#endif

#endif /* _DLFCN_H_ */
