/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 14, 2024.
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
/* This work is part of OpenLDAP Software <http://www.openldap.org/>.
 *
 * Copyright 1998-2011 The OpenLDAP Foundation.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted only as authorized by the OpenLDAP
 * Public License.
 *
 * A copy of this license is available in the file LICENSE in the
 * top-level directory of the distribution or, alternatively, at
 * <http://www.OpenLDAP.org/license.html>.
 */

/*
 * File Locking Routines
 *
 * Implementations (in order of preference)
 *	- lockf
 *	- fcntl
 *  - flock
 *
 * Other implementations will be added as needed.
 *
 * NOTE: lutil_lockf() MUST block until an exclusive lock is acquired.
 */

#include "portable.h"

#include <stdio.h>
#include <ac/unistd.h>

#undef LOCK_API

#if defined(HAVE_LOCKF) && defined(F_LOCK)
#	define USE_LOCKF 1
#	define LOCK_API	"lockf"
#endif

#if !defined(LOCK_API) && defined(HAVE_FCNTL)
#	ifdef HAVE_FCNTL_H
#		include <fcntl.h>
#	endif
#	ifdef F_WRLCK
#		define USE_FCNTL 1
#		define LOCK_API	"fcntl"
#	endif
#endif

#if !defined(LOCK_API) && defined(HAVE_FLOCK)
#	ifdef HAVE_SYS_FILE_H
#		include <sys/file.h>
#	endif
#	define USE_FLOCK 1
#	define LOCK_API	"flock"
#endif

#if !defined(USE_LOCKF) && !defined(USE_FCNTL) && !defined(USE_FLOCK)
int lutil_lockf ( int fd ) {
    fd = fd;
    return 0;
}

int lutil_unlockf ( int fd ) {
    fd = fd;
    return 0;
}
#endif

#ifdef USE_LOCKF
int lutil_lockf ( int fd ) {
	/* use F_LOCK instead of F_TLOCK, ie: block */
	return lockf( fd, F_LOCK, 0 );
}

int lutil_unlockf ( int fd ) {
	return lockf( fd, F_ULOCK, 0 );
}
#endif

#ifdef USE_FCNTL
int lutil_lockf ( int fd ) {
	struct flock file_lock;

	memset( &file_lock, '\0', sizeof( file_lock ) );
	file_lock.l_type = F_WRLCK;
	file_lock.l_whence = SEEK_SET;
	file_lock.l_start = 0;
	file_lock.l_len = 0;

	/* use F_SETLKW instead of F_SETLK, ie: block */
	return( fcntl( fd, F_SETLKW, &file_lock ) );
}

int lutil_unlockf ( int fd ) {
	struct flock file_lock;

	memset( &file_lock, '\0', sizeof( file_lock ) );
	file_lock.l_type = F_UNLCK;
	file_lock.l_whence = SEEK_SET;
	file_lock.l_start = 0;
	file_lock.l_len = 0;

	return( fcntl ( fd, F_SETLKW, &file_lock ) );
}
#endif

#ifdef USE_FLOCK
int lutil_lockf ( int fd ) {
	/* use LOCK_EX instead of LOCK_EX|LOCK_NB, ie: block */
	return flock( fd, LOCK_EX );
}

int lutil_unlockf ( int fd ) {
	return flock( fd, LOCK_UN );
}
#endif
