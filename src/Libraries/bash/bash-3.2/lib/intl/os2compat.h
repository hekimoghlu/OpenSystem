/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 8, 2023.
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
/* When included from os2compat.h we need all the original definitions */
#ifndef OS2_AWARE

#undef LIBDIR
#define LIBDIR			_nlos2_libdir
extern char *_nlos2_libdir;

#undef LOCALEDIR
#define LOCALEDIR		_nlos2_localedir
extern char *_nlos2_localedir;

#undef LOCALE_ALIAS_PATH
#define LOCALE_ALIAS_PATH	_nlos2_localealiaspath
extern char *_nlos2_localealiaspath;

#endif

#undef HAVE_STRCASECMP
#define HAVE_STRCASECMP 1
#define strcasecmp stricmp
#define strncasecmp strnicmp

/* We have our own getenv() which works even if library is compiled as DLL */
#define getenv _nl_getenv

/* Older versions of gettext used -1 as the value of LC_MESSAGES */
#define LC_MESSAGES_COMPAT (-1)
