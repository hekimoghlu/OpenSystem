/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 2, 2025.
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
#ifndef pcap_pcap_inttypes_h
#define pcap_pcap_inttypes_h

/*
 * If we're compiling with Visual Studio, make sure we have at least
 * VS 2015 or later, so we have sufficient C99 support.
 *
 * XXX - verify that we have at least C99 support on UN*Xes?
 *
 * What about MinGW or various DOS toolchains?  We're currently assuming
 * sufficient C99 support there.
 */
#if defined(_MSC_VER)
  /*
   * Compiler is MSVC.  Make sure we have VS 2015 or later.
   */
  #if _MSC_VER < 1900
    #error "Building libpcap requires VS 2015 or later"
  #endif
#endif

/*
 * Include <inttypes.h> to get the integer types and PRi[doux]64 values
 * defined.
 *
 * If the compiler is MSVC, we require VS 2015 or newer, so we
 * have <inttypes.h> - and support for %zu in the formatted
 * printing functions.
 *
 * If the compiler is MinGW, we assume we have <inttypes.h> - and
 * support for %zu in the formatted printing functions.
 *
 * If the target is UN*X, we assume we have a C99-or-later development
 * environment, and thus have <inttypes.h> - and support for %zu in
 * the formatted printing functions.
 *
 * If the target is MS-DOS, we assume we have <inttypes.h> - and support
 * for %zu in the formatted printing functions.
 *
 * I.e., assume we have <inttypes.h> and that it suffices.
 */

/*
 * XXX - somehow make sure we have enough C99 support with other
 * compilers and support libraries?
 */

#include <inttypes.h>

#endif /* pcap/pcap-inttypes.h */
