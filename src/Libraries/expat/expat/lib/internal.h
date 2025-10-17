/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 6, 2025.
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
#if defined(__GNUC__) && defined(__i386__) && ! defined(__MINGW32__)
/* We'll use this version by default only where we know it helps.

   regparm() generates warnings on Solaris boxes.   See SF bug #692878.

   Instability reported with egcs on a RedHat Linux 7.3.
   Let's comment out:
   #define FASTCALL __attribute__((stdcall, regparm(3)))
   and let's try this:
*/
#  define FASTCALL __attribute__((regparm(3)))
#  define PTRFASTCALL __attribute__((regparm(3)))
#endif

/* Using __fastcall seems to have an unexpected negative effect under
   MS VC++, especially for function pointers, so we won't use it for
   now on that platform. It may be reconsidered for a future release
   if it can be made more effective.
   Likely reason: __fastcall on Windows is like stdcall, therefore
   the compiler cannot perform stack optimizations for call clusters.
*/

/* Make sure all of these are defined if they aren't already. */

#ifndef FASTCALL
#  define FASTCALL
#endif

#ifndef PTRCALL
#  define PTRCALL
#endif

#ifndef PTRFASTCALL
#  define PTRFASTCALL
#endif

#ifndef XML_MIN_SIZE
#  if ! defined(__cplusplus) && ! defined(inline)
#    ifdef __GNUC__
#      define inline __inline
#    endif /* __GNUC__ */
#  endif
#endif /* XML_MIN_SIZE */

#ifdef __cplusplus
#  define inline inline
#else
#  ifndef inline
#    define inline
#  endif
#endif

#include <limits.h> // ULONG_MAX

#if defined(_WIN32)                                                            \
    && (! defined(__USE_MINGW_ANSI_STDIO)                                      \
        || (1 - __USE_MINGW_ANSI_STDIO - 1 == 0))
#  define EXPAT_FMT_ULL(midpart) "%" midpart "I64u"
#  if defined(_WIN64) // Note: modifiers "td" and "zu" do not work for MinGW
#    define EXPAT_FMT_PTRDIFF_T(midpart) "%" midpart "I64d"
#    define EXPAT_FMT_SIZE_T(midpart) "%" midpart "I64u"
#  else
#    define EXPAT_FMT_PTRDIFF_T(midpart) "%" midpart "d"
#    define EXPAT_FMT_SIZE_T(midpart) "%" midpart "u"
#  endif
#else
#  define EXPAT_FMT_ULL(midpart) "%" midpart "llu"
#  if ! defined(ULONG_MAX)
#    error Compiler did not define ULONG_MAX for us
#  elif ULONG_MAX == 18446744073709551615u // 2^64-1
#    define EXPAT_FMT_PTRDIFF_T(midpart) "%" midpart "ld"
#    define EXPAT_FMT_SIZE_T(midpart) "%" midpart "lu"
#  elif defined(EMSCRIPTEN) // 32bit mode Emscripten
#    define EXPAT_FMT_PTRDIFF_T(midpart) "%" midpart "ld"
#    define EXPAT_FMT_SIZE_T(midpart) "%" midpart "zu"
#  else
#    define EXPAT_FMT_PTRDIFF_T(midpart) "%" midpart "d"
#    define EXPAT_FMT_SIZE_T(midpart) "%" midpart "u"
#  endif
#endif

#ifndef UNUSED_P
#  define UNUSED_P(p) (void)p
#endif

/* NOTE BEGIN If you ever patch these defaults to greater values
              for non-attack XML payload in your environment,
              please file a bug report with libexpat.  Thank you!
*/
#define EXPAT_BILLION_LAUGHS_ATTACK_PROTECTION_MAXIMUM_AMPLIFICATION_DEFAULT   \
  100.0f
#define EXPAT_BILLION_LAUGHS_ATTACK_PROTECTION_ACTIVATION_THRESHOLD_DEFAULT    \
  8388608 // 8 MiB, 2^23
/* NOTE END */

#include "expat.h" // so we can use type XML_Parser below

#ifdef __cplusplus
extern "C" {
#endif

void _INTERNAL_trim_to_complete_utf8_characters(const char *from,
                                                const char **fromLimRef);

#if defined(XML_GE) && XML_GE == 1
unsigned long long testingAccountingGetCountBytesDirect(XML_Parser parser);
unsigned long long testingAccountingGetCountBytesIndirect(XML_Parser parser);
const char *unsignedCharToPrintable(unsigned char c);
#endif

extern
#if ! defined(XML_TESTING)
    const
#endif
    XML_Bool g_reparseDeferralEnabledDefault; // written ONLY in runtests.c
#if defined(XML_TESTING)
extern unsigned int g_bytesScanned; // used for testing only
#endif

#ifdef __cplusplus
}
#endif
