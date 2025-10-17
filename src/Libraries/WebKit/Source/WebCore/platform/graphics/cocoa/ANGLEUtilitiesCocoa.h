/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 27, 2023.
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

#if ENABLE(WEBGL)

#include "GraphicsTypesGL.h"
#include "IntSize.h"

typedef struct __IOSurface *IOSurfaceRef;
@protocol MTLSharedEvent;
@protocol MTLRasterizationRateMap;

namespace WebCore {

// Returns a handle which, if non-null, must be released with destroyPbufferAndDetachIOSurface().
void* createPbufferAndAttachIOSurface(GCGLDisplay, GCGLConfig, GCGLenum target, GCGLint usageHint, GCGLenum internalFormat, GCGLsizei width, GCGLsizei height, GCGLenum type, IOSurfaceRef, GCGLuint plane);

void destroyPbufferAndDetachIOSurface(GCGLDisplay, void* handle);

RetainPtr<id<MTLRasterizationRateMap>> newRasterizationRateMap(GCGLDisplay, IntSize, IntSize, IntSize, std::span<const float>, std::span<const float>, std::span<const float>);

RetainPtr<id<MTLSharedEvent>> newSharedEventWithMachPort(GCGLDisplay, mach_port_t);
RetainPtr<id<MTLSharedEvent>> newSharedEvent(GCGLDisplay);

}

#endif
