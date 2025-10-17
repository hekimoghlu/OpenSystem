/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 20, 2024.
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
#ifndef _CSSMCONFIG_H_
#define _CSSMCONFIG_H_  1

#include <AvailabilityMacros.h>
#include <TargetConditionals.h>
#include <ConditionalMacros.h>


/* #if defined(TARGET_API_MAC_OS8) || defined(TARGET_API_MAC_CARBON) || defined(TARGET_API_MAC_OSX) */
#if TARGET_OS_MAC
#include <sys/types.h>
#include <stdint.h>
#else
#error Unknown API architecture.
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifndef _SINT64
typedef int64_t sint64;
#define _SINT64
#endif
#ifndef _UINT64
typedef uint64_t uint64;
#define _UINT64
#endif
#ifndef _SINT32
typedef int32_t sint32;
#define _SINT32
#endif
#ifndef _SINT16
typedef int16_t sint16;
#define _SINT16
#endif
#ifndef _SINT8
typedef int8_t sint8;
#define _SINT8
#endif
#ifndef _UINT32
typedef uint32_t uint32;
#define _UINT32
#endif
#ifndef _UINT16
typedef uint16_t uint16;
#define _UINT16
#endif
#ifndef _UINT8
typedef uint8_t uint8;
#define _UINT8
#endif

typedef intptr_t CSSM_INTPTR;
typedef size_t CSSM_SIZE;

#define CSSMACI
#define CSSMAPI
#define CSSMCLI
#define CSSMCSPI
#define CSSMDLI
#define CSSMKRI
#define CSSMSPI
#define CSSMTPI

#ifdef __cplusplus
}
#endif

#endif /* _CSSMCONFIG_H_ */
