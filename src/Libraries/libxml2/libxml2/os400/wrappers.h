/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 6, 2021.
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
#ifndef __WRAPPERS_H_
#define __WRAPPERS_H_

/**
***     OS/400 specific defines.
**/

#define __cplusplus__strings__

/**
***     Force header inclusions before renaming procedures to UTF-8 wrappers.
**/

#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>

#include "dlfcn.h"


/**
***     UTF-8 wrappers prototypes.
**/

extern int      _lx_getaddrinfo(const char * node, const char * service,
                        const struct addrinfo * hints, struct addrinfo * * res);
extern const char *
                _lx_inet_ntop(int af,
                        const void * src, char * dst, socklen_t size);
extern void *   _lx_dlopen(const char * filename, int flag);
extern void *   _lx_dlsym(void * handle, const char * symbol);
extern char *   _lx_dlerror(void);


#ifdef LIBXML_ZLIB_ENABLED

#include <zlib.h>

extern gzFile   _lx_gzopen(const char * path, const char * mode);
extern gzFile   _lx_gzdopen(int fd, const char * mode);

#endif


/**
***     Rename data/procedures to UTF-8 wrappers.
**/

#define getaddrinfo     _lx_getaddrinfo
#define inet_ntop       _lx_inet_ntop
#define dlopen          _lx_dlopen
#define dlsym           _lx_dlsym
#define dlerror         _lx_dlerror
#define gzopen          _lx_gzopen
#define gzdopen         _lx_gzdopen
#define inflateInit2_   _lx_inflateInit2_
#define deflateInit2_   _lx_deflateInit2_

#endif
