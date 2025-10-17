/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 3, 2022.
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
/* expat_config.h.in.  Generated from configure.ac by autoheader.  */

#ifndef EXPAT_CONFIG_H
#define EXPAT_CONFIG_H 1

/* Define if building universal (internal helper macro) */
/* #undef AC_APPLE_UNIVERSAL_BUILD */

/* 1234 = LILENDIAN, 4321 = BIGENDIAN */
#ifdef __BIG_ENDIAN__
#define BYTEORDER 4321
#else
#define BYTEORDER 1234
#endif

/* Define to 1 if you have the `arc4random' function. */
/* #undef HAVE_ARC4RANDOM */

/* Define to 1 if you have the `arc4random_buf' function. */
#define HAVE_ARC4RANDOM_BUF 1

/* define if the compiler supports basic C++11 syntax */
#define HAVE_CXX11 1

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* Define to 1 if you have the <fcntl.h> header file. */
#define HAVE_FCNTL_H 1

/* Define to 1 if you have the `getpagesize' function. */
#define HAVE_GETPAGESIZE 1

/* Define to 1 if you have the `getrandom' function. */
/* #undef HAVE_GETRANDOM */

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the `bsd' library (-lbsd). */
/* #undef HAVE_LIBBSD */

/* Define to 1 if you have a working `mmap' system call. */
#define HAVE_MMAP 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdio.h> header file. */
#define HAVE_STDIO_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have `syscall' and `SYS_getrandom'. */
/* #undef HAVE_SYSCALL_GETRANDOM */

/* Define to 1 if you have the <sys/param.h> header file. */
#define HAVE_SYS_PARAM_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define to the sub-directory where libtool stores uninstalled libraries. */
#define LT_OBJDIR ".libs/"

/* Name of package */
#define PACKAGE "expat"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "https://github.com/libexpat/libexpat/issues"

/* Define to the full name of this package. */
#define PACKAGE_NAME "expat"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "expat 2.6.3"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "expat"

/* Define to the home page for this package. */
#define PACKAGE_URL ""

/* Define to the version of this package. */
#define PACKAGE_VERSION "2.6.3"

/* Define to 1 if all of the C90 standard headers exist (not just the ones
   required in a freestanding environment). This macro is provided for
   backward compatibility; new code need not use it. */
#define STDC_HEADERS 1

/* Version number of package */
#define VERSION "2.6.3"

/* Define WORDS_BIGENDIAN to 1 if your processor stores words with the most
   significant byte first (like Motorola and SPARC, unlike Intel). */
#if defined AC_APPLE_UNIVERSAL_BUILD
# if defined __BIG_ENDIAN__
#  define WORDS_BIGENDIAN 1
# endif
#else
# ifndef WORDS_BIGENDIAN
/* #  undef WORDS_BIGENDIAN */
# endif
#endif

/* Define to allow retrieving the byte offsets for attribute names and values.
   */
/* #undef XML_ATTR_INFO */

/* Define to specify how much context to retain around the current parse
   point, 0 to disable. */
#define XML_CONTEXT_BYTES 1024

/* Define to include code reading entropy from `/dev/urandom'. */
#define XML_DEV_URANDOM 1

/* Define to make parameter entity parsing functionality available. */
#define XML_DTD 1

/* Define as 1/0 to enable/disable support for general entities. */
#define XML_GE 1

/* Define to make XML Namespaces functionality available. */
#define XML_NS 1

/* Define to empty if `const' does not conform to ANSI C. */
/* #undef const */

/* Define to `long int' if <sys/types.h> does not define. */
/* #undef off_t */

#endif // ndef EXPAT_CONFIG_H
