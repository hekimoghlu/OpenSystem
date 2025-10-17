/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 21, 2024.
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
#ifndef _IOKIT_IOGRAPHICSINTERFACE_H
#define _IOKIT_IOGRAPHICSINTERFACE_H

#ifdef KERNEL
#define NO_CFPLUGIN     1
#endif

#ifndef NO_CFPLUGIN
#include <IOKit/IOCFPlugIn.h>
#endif /* ! NO_CFPLUGIN */

#define IOGA_COMPAT     1

#include <IOKit/graphics/IOGraphicsInterfaceTypes.h>

// <rdar://problem/23764215> IOGraphics: IOGraphicsInterface.h: "C" linkage not enforced.
#ifdef __cplusplus
extern "C" {
#endif

#define kIOGraphicsAcceleratorTypeID                    \
        (CFUUIDGetConstantUUIDWithBytes(NULL,           \
                                0xAC, 0xCF, 0x00, 0x00, \
                                0x00, 0x00,             \
                                0x00, 0x00,             \
                                0x00, 0x00,             \
                                0x00, 0x0a, 0x27, 0x89, 0x90, 0x4e))

// IOGraphicsAcceleratorType objects must implement the
// IOGraphicsAcceleratorInterface

#define kIOGraphicsAcceleratorInterfaceID               \
        (CFUUIDGetConstantUUIDWithBytes(NULL,           \
                                0x67, 0x66, 0xE9, 0x4A, \
                                0x00, 0x00,             \
                                0x00, 0x00,             \
                                0x00, 0x00,             \
                                0x00, 0x0a, 0x27, 0x89, 0x90, 0x4e))

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

typedef
IOReturn (*IOBlitAccumulatePtr)(void *thisPointer,
                                SInt32 a, SInt32 b, SInt32 c,
                                SInt32 d, SInt32 e, SInt32 f );

#ifdef IOGA_COMPAT
typedef
IOReturn (*IOBlitProcPtr)(void *thisPointer,
                          IOOptionBits options,
                          IOBlitType type, IOBlitSourceDestType sourceDestType,
                          IOBlitOperation * operation,
                          void * source, void * destination,
                          IOBlitCompletionToken * completionToken );
#endif

typedef
IOReturn (*IOBlitterPtr)(void *thisPointer,
                          IOOptionBits options,
                          IOBlitType type, IOBlitSourceType sourceType,
                          IOBlitOperation * operation,
                          void * source );

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef NO_CFPLUGIN

typedef struct IOGraphicsAcceleratorInterfaceStruct {
    IUNKNOWN_C_GUTS;
    IOCFPLUGINBASE;
    
    IOReturn (*Reset)
        (void *thisPointer, IOOptionBits options);
    IOReturn (*CopyCapabilities)
        (void *thisPointer, FourCharCode select, CFTypeRef * capabilities);
#ifdef IOGA_COMPAT
    IOReturn (*GetBlitProc)
        (void *thisPointer, IOOptionBits options,
         IOBlitType type, IOBlitSourceDestType sourceDestType,
         IOBlitProcPtr * blitProc );
#else
    void * __gaInterfaceReserved0;
#endif

    IOReturn (*Flush)
        (void *thisPointer, IOOptionBits options);
#ifdef IOGA_COMPAT
    IOReturn (*WaitForCompletion)
        (void *thisPointer, IOOptionBits options,
         IOBlitCompletionToken completionToken);
#else
    void * __gaInterfaceReserved1;
#endif
    IOReturn (*Synchronize)
        (void *thisPointer, UInt32 options,
         UInt32 x, UInt32 y, UInt32 w, UInt32 h );
    IOReturn (*GetBeamPosition)
        (void *thisPointer, IOOptionBits options, SInt32 * position);
    IOReturn (*AllocateSurface)
        (void *thisPointer, IOOptionBits options,
         IOBlitSurface * surface, void * cgsSurfaceID );
    IOReturn (*FreeSurface)
        (void *thisPointer, IOOptionBits options, IOBlitSurface * surface);
    IOReturn (*LockSurface)
        (void *thisPointer, IOOptionBits options, IOBlitSurface * surface,
         vm_address_t * address );
    IOReturn (*UnlockSurface)
        (void *thisPointer, IOOptionBits options, IOBlitSurface * surface,
         IOOptionBits * swapFlags);
    IOReturn (*SwapSurface)
        (void *thisPointer, IOOptionBits options,
         IOBlitSurface * surface, IOOptionBits * swapFlags);

    IOReturn (*SetDestination)
        (void *thisPointer, IOOptionBits options, IOBlitSurface * surface );

    IOReturn (*GetBlitter)
        (void *thisPointer, IOOptionBits options,
         IOBlitType type, IOBlitSourceType sourceType,
         IOBlitterPtr * blitter );
    IOReturn (*WaitComplete)
        (void *thisPointer, IOOptionBits options);

    void * __gaInterfaceReserved[ 24 ];

} IOGraphicsAcceleratorInterface;

#endif /* ! NO_CFPLUGIN */

/* Helper function for plugin use */
IOReturn IOAccelFindAccelerator( io_service_t framebuffer,
                                 io_service_t * pAccelerator, UInt32 * pFramebufferIndex );


#ifdef __cplusplus
    }
#endif

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#endif /* !_IOKIT_IOGRAPHICSINTERFACE_H */
