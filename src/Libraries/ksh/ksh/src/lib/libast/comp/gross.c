/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 21, 2023.
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
 * porting hacks here
 */

#include <ast.h>
#include <ls.h>

#include "FEATURE/hack"

void _STUB_gross(){}

#if _lcl_xstat

extern int fstat(int fd, struct stat* st)
{
#if _lib___fxstat
	return __fxstat(_STAT_VER, fd, st);
#else
	return _fxstat(_STAT_VER, fd, st);
#endif
}

extern int lstat(const char* path, struct stat* st)
{
#if _lib___lxstat
	return __lxstat(_STAT_VER, path, st);
#else
	return _lxstat(_STAT_VER, path, st);
#endif
}

extern int stat(const char* path, struct stat* st)
{
#if _lib___xstat
	return __xstat(_STAT_VER, path, st);
#else
	return _xstat(_STAT_VER, path, st);
#endif
}

#endif

#if _lcl_xstat64

extern int fstat64(int fd, struct stat64* st)
{
#if _lib___fxstat64
	return __fxstat64(_STAT_VER, fd, st);
#else
	return _fxstat64(_STAT_VER, fd, st);
#endif
}

extern int lstat64(const char* path, struct stat64* st)
{
#if _lib___lxstat64
	return __lxstat64(_STAT_VER, path, st);
#else
	return _lxstat64(_STAT_VER, path, st);
#endif
}

extern int stat64(const char* path, struct stat64* st)
{
#if _lib___xstat64
	return __xstat64(_STAT_VER, path, st);
#else
	return _xstat64(_STAT_VER, path, st);
#endif
}

#endif

#if __sgi && _hdr_locale_attr

#include "gross_sgi.h"

#endif
