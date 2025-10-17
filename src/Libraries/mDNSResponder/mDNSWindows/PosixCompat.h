/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 13, 2023.
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
#pragma once

#include "CommonServices.h"
#include <winsock2.h>
#include <time.h>


/* 
 * Posix process compatibility
 */
typedef int pid_t;
#if !defined( getpid )
#	define getpid _getpid
#endif


/* 
 * Posix networking compatibility
 */
extern unsigned
if_nametoindex( const char * ifname );


extern char*
if_indextoname( unsigned ifindex, char * ifname );


/* 
 * Posix time compatibility
 */
extern int
gettimeofday( struct timeval * tv, struct timezone * tz );


extern struct tm*
localtime_r( const time_t * clock, struct tm * result );


/* 
 * Posix string compatibility
 */
#if !defined( strcasecmp )
#	define strcasecmp	_stricmp
#endif

#if !defined( snprintf )
#	define snprint		_snprintf
#endif

