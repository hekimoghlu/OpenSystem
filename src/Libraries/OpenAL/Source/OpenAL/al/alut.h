/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 27, 2025.
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
#ifndef _ALUT_H_
#define _ALUT_H_

#define ALUTAPI
#define ALUTAPIENTRY

#include <OpenAL/al.h>

#ifdef __cplusplus
extern "C" {
#endif
   
#ifdef TARGET_OS_MAC
   #if TARGET_OS_MAC
       #pragma export on
   #endif
#endif

ALUTAPI ALvoid	ALUTAPIENTRY alutInit(ALint *argc,ALbyte **argv) OPENAL_DEPRECATED;
ALUTAPI ALvoid	ALUTAPIENTRY alutExit(ALvoid) OPENAL_DEPRECATED;
ALUTAPI ALvoid	ALUTAPIENTRY alutLoadWAVFile(ALbyte *file,ALenum *format,ALvoid **data,ALsizei *size,ALsizei *freq) OPENAL_DEPRECATED;
ALUTAPI ALvoid  ALUTAPIENTRY alutLoadWAVMemory(ALbyte *memory,ALenum *format,ALvoid **data,ALsizei *size,ALsizei *freq) OPENAL_DEPRECATED;
ALUTAPI ALvoid  ALUTAPIENTRY alutUnloadWAV(ALenum format,ALvoid *data,ALsizei size,ALsizei freq) OPENAL_DEPRECATED;

#ifdef TARGET_OS_MAC
   #if TARGET_OS_MAC
      #pragma export off
   #endif
#endif

#ifdef __cplusplus
}
#endif

#endif
