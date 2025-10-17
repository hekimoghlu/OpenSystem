/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 20, 2021.
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
#pragma once

#include <CoreFoundation/CoreFoundation.h>

#if USE(APPLE_INTERNAL_SDK)

// FIXME: Remove once rdar://131328679 is fixed and distributed.
IGNORE_WARNINGS_BEGIN("#warnings")
#include <AccessibilitySupport.h>
IGNORE_WARNINGS_END

#else

typedef CF_ENUM(int32_t, AXSIsolatedTreeMode)
{
    AXSIsolatedTreeModeOff = 0,
    AXSIsolatedTreeModeMainThread,
    AXSIsolatedTreeModeSecondaryThread,
};

#endif

WTF_EXTERN_C_BEGIN

AXSIsolatedTreeMode _AXSIsolatedTreeMode(void);
void _AXSSetIsolatedTreeMode(AXSIsolatedTreeMode);

extern CFStringRef kAXSEnhanceTextLegibilityChangedNotification;
Boolean _AXSEnhanceTextLegibilityEnabled();

WTF_EXTERN_C_END
