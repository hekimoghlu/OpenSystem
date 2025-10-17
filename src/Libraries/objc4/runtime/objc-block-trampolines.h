/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 12, 2023.
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
#ifndef _OBJC_TRAMPOLINES_H
#define _OBJC_TRAMPOLINES_H

/* 
 * WARNING  DANGER  HAZARD  BEWARE  EEK
 * 
 * Everything in this file is for Apple Internal use only.
 * These will change in arbitrary OS updates and in unpredictable ways.
 * When your program breaks, you get to keep both pieces.
 */

/*
 * objc-block-trampolines.h: Symbols for IMP block trampolines
 */

// WARNING: remapped code and dtrace do not play well together. Dtrace
// will place trap instructions to instrument the code, which then get
// remapped along with everything else. The remapped traps are not
// recognized by dtrace and the process crashes. To avoid this, dtrace
// blacklists this library by name. Do not change the name of this
// library. rdar://problem/42627391

#include <TargetConditionals.h>

#if !TARGET_OS_EXCLAVEKIT

#include <objc/objc-api.h>

OBJC_EXPORT const char _objc_blockTrampolineImpl
    OBJC_AVAILABLE(10.14, 12.0, 12.0, 5.0, 3.0);

OBJC_EXPORT const char _objc_blockTrampolineStart
    OBJC_AVAILABLE(10.14, 12.0, 12.0, 5.0, 3.0);

OBJC_EXPORT const char _objc_blockTrampolineLast
    OBJC_AVAILABLE(10.14, 12.0, 12.0, 5.0, 3.0);


OBJC_EXPORT const char _objc_blockTrampolineImpl_stret
OBJC_AVAILABLE(10.14, 12.0, 12.0, 5.0, 3.0)
    OBJC_ARM64_UNAVAILABLE;

OBJC_EXPORT const char _objc_blockTrampolineStart_stret
OBJC_AVAILABLE(10.14, 12.0, 12.0, 5.0, 3.0)
    OBJC_ARM64_UNAVAILABLE;

OBJC_EXPORT const char _objc_blockTrampolineLast_stret
OBJC_AVAILABLE(10.14, 12.0, 12.0, 5.0, 3.0)
    OBJC_ARM64_UNAVAILABLE;

#endif // !TARGET_OS_EXCLAVEKIT

#endif
