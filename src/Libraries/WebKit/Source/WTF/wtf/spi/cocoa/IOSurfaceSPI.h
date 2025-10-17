/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 11, 2024.
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

#if HAVE(IOSURFACE)

#if PLATFORM(MAC) || USE(APPLE_INTERNAL_SDK)

#include <IOSurface/IOSurface.h>

#else

#include <CoreFoundation/CFBase.h>
#include <mach/mach_port.h>
#include <wtf/spi/cocoa/IOReturnSPI.h>
#include <wtf/spi/cocoa/IOTypesSPI.h>

typedef struct __IOSurface *IOSurfaceRef;

#endif

#ifdef __OBJC__
#import <IOSurface/IOSurfaceObjC.h>
#endif

WTF_EXTERN_C_BEGIN

extern const CFStringRef kIOSurfaceAllocSize;
extern const CFStringRef kIOSurfaceBytesPerElement;
extern const CFStringRef kIOSurfaceBytesPerRow;
extern const CFStringRef kIOSurfaceCacheMode;
extern const CFStringRef kIOSurfaceColorSpace;
extern const CFStringRef kIOSurfaceHeight;
extern const CFStringRef kIOSurfacePixelFormat;
extern const CFStringRef kIOSurfaceWidth;
extern const CFStringRef kIOSurfaceElementWidth;
extern const CFStringRef kIOSurfaceElementHeight;
extern const CFStringRef kIOSurfaceName;
extern const CFStringRef kIOSurfacePlaneWidth;
extern const CFStringRef kIOSurfacePlaneHeight;
extern const CFStringRef kIOSurfacePlaneBytesPerRow;
extern const CFStringRef kIOSurfacePlaneOffset;
extern const CFStringRef kIOSurfacePlaneSize;
extern const CFStringRef kIOSurfacePlaneInfo;

size_t IOSurfaceAlignProperty(CFStringRef property, size_t value);
IOSurfaceRef IOSurfaceCreate(CFDictionaryRef properties);
mach_port_t IOSurfaceCreateMachPort(IOSurfaceRef buffer);
size_t IOSurfaceGetAllocSize(IOSurfaceRef buffer);
void *IOSurfaceGetBaseAddress(IOSurfaceRef buffer);
size_t IOSurfaceGetBytesPerRow(IOSurfaceRef buffer);
size_t IOSurfaceGetHeight(IOSurfaceRef buffer);
size_t IOSurfaceGetPropertyMaximum(CFStringRef property);
size_t IOSurfaceGetWidth(IOSurfaceRef buffer);
OSType IOSurfaceGetPixelFormat(IOSurfaceRef buffer);
void IOSurfaceIncrementUseCount(IOSurfaceRef buffer);
Boolean IOSurfaceIsInUse(IOSurfaceRef buffer);
IOReturn IOSurfaceLock(IOSurfaceRef buffer, uint32_t options, uint32_t *seed);
IOSurfaceRef IOSurfaceLookupFromMachPort(mach_port_t);
IOReturn IOSurfaceUnlock(IOSurfaceRef buffer, uint32_t options, uint32_t *seed);
size_t IOSurfaceGetWidthOfPlane(IOSurfaceRef buffer, size_t planeIndex);
size_t IOSurfaceGetHeightOfPlane(IOSurfaceRef buffer, size_t planeIndex);

WTF_EXTERN_C_END

#if USE(APPLE_INTERNAL_SDK)

#import <IOSurface/IOSurfacePrivate.h>

#else

#import <IOSurface/IOSurfaceTypes.h>

WTF_EXTERN_C_BEGIN

#if (HAVE(IOSURFACE_SET_OWNERSHIP) || HAVE(IOSURFACE_SET_OWNERSHIP_IDENTITY)) && !HAVE(BROWSER_ENGINE_SUPPORTING_API)
typedef CF_ENUM(int, IOSurfaceMemoryLedgerTags)
{
    kIOSurfaceMemoryLedgerTagDefault     = 0x00000001,
    kIOSurfaceMemoryLedgerTagNetwork     = 0x00000002,
    kIOSurfaceMemoryLedgerTagMedia       = 0x00000003,
    kIOSurfaceMemoryLedgerTagGraphics    = 0x00000004,
    kIOSurfaceMemoryLedgerTagNeural      = 0x00000005,
};
#endif

#if HAVE(IOSURFACE_SET_OWNERSHIP)
IOReturn IOSurfaceSetOwnership(IOSurfaceRef buffer, task_t newOwner, int newLedgerTag, uint32_t newLedgerOptions);
#endif

#if HAVE(IOSURFACE_SET_OWNERSHIP_IDENTITY)
kern_return_t IOSurfaceSetOwnershipIdentity(IOSurfaceRef buffer, mach_port_t task_id_token, int newLedgerTag, uint32_t newLedgerOptions);
#endif

WTF_EXTERN_C_END

#endif

WTF_EXTERN_C_BEGIN

IOReturn IOSurfaceSetPurgeable(IOSurfaceRef buffer, uint32_t newState, uint32_t *oldState);

WTF_EXTERN_C_END

#if HAVE(IOSURFACE_ACCELERATOR)
#if USE(APPLE_INTERNAL_SDK)

#import <IOSurfaceAccelerator/IOSurfaceAcceleratorTypes.h>

// Workaround for <rdar://105279275>.
#ifndef MSR_USE_SHARED_EVENT
#define MSR_USE_SHARED_EVENT 0
#endif

#import <IOSurfaceAccelerator/IOSurfaceAccelerator.h>

#else

typedef struct __IOSurfaceAccelerator *IOSurfaceAcceleratorRef;

WTF_EXTERN_C_BEGIN

extern const CFStringRef kIOSurfaceAcceleratorUnwireSurfaceKey;

IOReturn IOSurfaceAcceleratorCreate(CFAllocatorRef, CFDictionaryRef properties, IOSurfaceAcceleratorRef* acceleratorOut);
CFRunLoopSourceRef IOSurfaceAcceleratorGetRunLoopSource(IOSurfaceAcceleratorRef);

typedef void (*IOSurfaceAcceleratorCompletionCallback)(void* completionRefCon, IOReturn status, void* completionRefCon2);

typedef struct IOSurfaceAcceleratorCompletion {
    IOSurfaceAcceleratorCompletionCallback completionCallback;
    void* completionRefCon;
    void* completionRefCon2;
} IOSurfaceAcceleratorCompletion;

IOReturn IOSurfaceAcceleratorTransformSurface(IOSurfaceAcceleratorRef, IOSurfaceRef sourceBuffer, IOSurfaceRef destinationBuffer, CFDictionaryRef options, void* pCropRectangles, IOSurfaceAcceleratorCompletion* pCompletion, void* pSwap, uint32_t* pCommandID);

WTF_EXTERN_C_END

#endif // USE(APPLE_INTERNAL_SDK)
#endif // HAVE(IOSURFACE_ACCELERATOR)

#endif // HAVE(IOSURFACE)
