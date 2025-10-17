/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 23, 2025.
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
#import <Foundation/Foundation.h>

#if USE(APPLE_INTERNAL_SDK)

// FIXME: Remove EAGL_IOSURFACE macro once header refactoring is completed.
#define EAGL_IOSURFACE 1
#import <OpenGLES/EAGLPrivate.h>
#undef EAGL_IOSURFACE

#else

typedef uint32_t EAGLContextParameter;

enum {
    kEAGLCPGPURestartStatusNone        = 0, /* context has not caused recent GPU restart */
    kEAGLCPGPURestartStatusCaused      = 1, /* context caused recent GPU restart (clear on query) */
    kEAGLCPGPURestartStatusBlacklisted = 2, /* context is being ignored for excessive GPU restarts */
};

/* (read-only) context caused GPU hang/crash, requiring a restart (see EAGLGPGPURestartStatus) */
#define kEAGLCPGPURestartStatus                   ((EAGLContextParameter)317)
/* (read-write) how to react to being blacklisted for causing excessive restarts (default to 1) */
#define kEAGLCPAbortOnGPURestartStatusBlacklisted ((EAGLContextParameter)318)
/* (read-only) does driver support auto-restart of GPU on hang/crash? */
#define kEAGLCPSupportGPURestart                  ((EAGLContextParameter)319)
/* (read-only) does driver/GPU support separate address space per context? */
#define kEAGLCPSupportSeparateAddressSpace        ((EAGLContextParameter)320)

@interface EAGLContext (EAGLPrivate)
- (NSUInteger)setParameter:(EAGLContextParameter)pname to:(int32_t *)params;
- (NSUInteger)getParameter:(EAGLContextParameter)pname to:(int32_t *)params;
@end
#endif
