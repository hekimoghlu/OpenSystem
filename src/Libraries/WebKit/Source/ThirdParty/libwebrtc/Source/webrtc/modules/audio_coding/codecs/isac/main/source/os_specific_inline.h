/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 30, 2025.
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
#ifndef MODULES_AUDIO_CODING_CODECS_ISAC_MAIN_SOURCE_OS_SPECIFIC_INLINE_H_
#define MODULES_AUDIO_CODING_CODECS_ISAC_MAIN_SOURCE_OS_SPECIFIC_INLINE_H_

#include <math.h>

#include "rtc_base/system/arch.h"

#if defined(WEBRTC_POSIX)
#define WebRtcIsac_lrint lrint
#elif (defined(WEBRTC_ARCH_X86) && defined(WIN32))
static __inline long int WebRtcIsac_lrint(double x_dbl) {
  long int x_int;

  __asm {
    fld x_dbl
    fistp x_int
  }
  ;

  return x_int;
}
#else  // Do a slow but correct implementation of lrint

static __inline long int WebRtcIsac_lrint(double x_dbl) {
  long int x_int;
  x_int = (long int)floor(x_dbl + 0.499999999999);
  return x_int;
}

#endif

#endif  // MODULES_AUDIO_CODING_CODECS_ISAC_MAIN_SOURCE_OS_SPECIFIC_INLINE_H_
