/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 25, 2024.
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
#ifndef portability_h
#define	portability_h

/*
 * Helpers for portability between Windows and UN*X and between different
 * flavors of UN*X.
 */
#include <stdarg.h>	/* we declare varargs functions on some platforms */

#include "pcap/funcattrs.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef HAVE_STRLCAT
  #define pcap_strlcat	strlcat
#else
  #if defined(_MSC_VER) || defined(__MINGW32__)
    /*
     * strncat_s() is supported at least back to Visual
     * Studio 2005; we require Visual Studio 2015 or later.
     */
    #define pcap_strlcat(x, y, z) \
	strncat_s((x), (z), (y), _TRUNCATE)
  #else
    /*
     * Define it ourselves.
     */
    extern size_t pcap_strlcat(char * restrict dst, const char * restrict src, size_t dstsize);
  #endif
#endif

#ifdef HAVE_STRLCPY
  #define pcap_strlcpy	strlcpy
#else
  #if defined(_MSC_VER) || defined(__MINGW32__)
    /*
     * strncpy_s() is supported at least back to Visual
     * Studio 2005; we require Visual Studio 2015 or later.
     */
    #define pcap_strlcpy(x, y, z) \
	strncpy_s((x), (z), (y), _TRUNCATE)
  #else
    /*
     * Define it ourselves.
     */
    extern size_t pcap_strlcpy(char * restrict dst, const char * restrict src, size_t dstsize);
  #endif
#endif

#ifdef _MSC_VER
  /*
   * If <crtdbg.h> has been included, and _DEBUG is defined, and
   * __STDC__ is zero, <crtdbg.h> will define strdup() to call
   * _strdup_dbg().  So if it's already defined, don't redefine
   * it.
   */
  #ifndef strdup
  #define strdup	_strdup
  #endif
#endif

/*
 * We want asprintf(), for some cases where we use it to construct
 * dynamically-allocated variable-length strings; it's present on
 * some, but not all, platforms.
 */
#ifdef HAVE_ASPRINTF
#define pcap_asprintf asprintf
#else
extern int pcap_asprintf(char **, PCAP_FORMAT_STRING(const char *), ...)
    PCAP_PRINTFLIKE(2, 3);
#endif

#ifdef HAVE_VASPRINTF
#define pcap_vasprintf vasprintf
#else
extern int pcap_vasprintf(char **, const char *, va_list ap);
#endif

#ifdef __APPLE__
/* To silence compiler warning about redefintion of timeradd and timersub */
#include <sys/time.h>
#endif /* __APPLE__ */

/* For Solaris before 11. */
#ifndef timeradd
#define timeradd(a, b, result)                       \
  do {                                               \
    (result)->tv_sec = (a)->tv_sec + (b)->tv_sec;    \
    (result)->tv_usec = (a)->tv_usec + (b)->tv_usec; \
    if ((result)->tv_usec >= 1000000) {              \
      ++(result)->tv_sec;                            \
      (result)->tv_usec -= 1000000;                  \
    }                                                \
  } while (0)
#endif /* timeradd */
#ifndef timersub
#define timersub(a, b, result)                       \
  do {                                               \
    (result)->tv_sec = (a)->tv_sec - (b)->tv_sec;    \
    (result)->tv_usec = (a)->tv_usec - (b)->tv_usec; \
    if ((result)->tv_usec < 0) {                     \
      --(result)->tv_sec;                            \
      (result)->tv_usec += 1000000;                  \
    }                                                \
  } while (0)
#endif /* timersub */

#ifdef HAVE_STRTOK_R
  #define pcap_strtok_r	strtok_r
#else
  #ifdef _WIN32
    /*
     * Microsoft gives it a different name.
     */
    #define pcap_strtok_r	strtok_s
  #else
    /*
     * Define it ourselves.
     */
    extern char *pcap_strtok_r(char *, const char *, char **);
  #endif
#endif /* HAVE_STRTOK_R */

#ifdef _WIN32
  #if !defined(__cplusplus)
    #define inline __inline
  #endif
#endif /* _WIN32 */

#ifdef __cplusplus
}
#endif

#endif
