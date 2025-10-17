/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 28, 2024.
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
#ifndef CONFIG_ZKT_H
# define CONFIG_ZKT_H

/* don't change anything below this */
/* the values here are determined or settable via the ./configure script */

#ifndef HAS_UTYPES
# define	HAS_UTYPES	1
#endif

/* # define	HAVE_TIMEGM		1	*/
/* # define	HAVE_GETOPT_LONG	1	*/
/* # define	HAVE_STRFTIME		1	*/

#ifndef COLOR_MODE
# define	COLOR_MODE	1
#endif

#ifndef TTL_IN_KEYFILE_ALLOWED
# define	TTL_IN_KEYFILE_ALLOWED	1
#endif

#ifndef PRINT_TIMEZONE
# define	PRINT_TIMEZONE	0
#endif

#ifndef PRINT_AGE_WITH_YEAR
# define	PRINT_AGE_WITH_YEAR	0
#endif

#ifndef LOG_WITH_PROGNAME
# define	LOG_WITH_PROGNAME	0
#endif

#ifndef LOG_WITH_TIMESTAMP
# define	LOG_WITH_TIMESTAMP	1
#endif

#ifndef LOG_WITH_LEVEL
# define	LOG_WITH_LEVEL		1
#endif

#ifndef ALWAYS_CHECK_KEYSETFILES
# define	ALWAYS_CHECK_KEYSETFILES	1
#endif

#ifndef ALLOW_ALWAYS_PREPUBLISH_ZSK
# define	ALLOW_ALWAYS_PREPUBLISH_ZSK	1
#endif

#ifndef CONFIG_PATH
# define	CONFIG_PATH	"/var/named/"
#endif

/* tree usage is setable by configure script parameter */
#ifndef USE_TREE
# define	USE_TREE	1
#endif

/* BIND version and utility path *must* be set by ./configure script */
#ifndef BIND_UTIL_PATH
# error ("BIND_UTIL_PATH not set. Please run configure with --enable-bind_util_path=");
#endif
#ifndef BIND_VERSION
# define	BIND_VERSION	980
#endif

#ifndef ZKT_VERSION
# if defined(USE_TREE) && USE_TREE
#  define	ZKT_VERSION	"vT1.1.0 (c) Feb 2005 - Jan 2012 Holger Zuleger hznet.de"
# else
#  define	ZKT_VERSION	"v1.1.0 (c) Feb 2005 - Jan 2012 Holger Zuleger hznet.de"
# endif
#endif


#if !defined(HAS_UTYPES) || !HAS_UTYPES
typedef	unsigned long	ulong;
typedef	unsigned int	uint;
typedef	unsigned short	ushort;
typedef	unsigned char	uchar;
#endif

#endif
