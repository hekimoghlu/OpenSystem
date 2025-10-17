/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 26, 2025.
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

#if USE(APPLE_INTERNAL_SDK)

#include <Carbon/CarbonPriv.h>

#else

#define kTSMInputSourcePropertyScriptCode CFSTR("TSMInputSourcePropertyScriptCode")

#endif

typedef struct __TSMInputSource* TSMInputSourceRef;
typedef CFStringRef TSMInputSourcePropertyTag;
typedef struct OpaqueEventRef* EventRef;
typedef OSType EventParamName;
typedef OSType EventParamType;

WTF_EXTERN_C_BEGIN

OSStatus _SyncWindowWithCGAfterMove(WindowRef);
CGWindowID GetNativeWindowFromWindowRef(WindowRef);
OSStatus TSMProcessRawKeyEvent(EventRef);
EventRef GetCurrentEvent();
CFTypeRef TSMGetInputSourceProperty(TSMInputSourceRef, TSMInputSourcePropertyTag);
OSStatus GetEventParameter(EventRef, EventParamName inName, EventParamType inDesiredType, EventParamType* outActualType, ByteCount inBufferSize, ByteCount* outActualSize, void* outData);

WTF_EXTERN_C_END
