/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 8, 2023.
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
#ifndef _IOACCEL_SURFACE_CONNECT_H
#define _IOACCEL_SURFACE_CONNECT_H

#include <IOKit/graphics/IOAccelTypes.h>
#include <IOKit/graphics/IOAccelClientConnect.h>

/*
** Surface visible region in device coordinates.
**
** num_rects:   The number of rectangles in the rect array.  If num_rects
**              is zero the bounds rectangle is used for the visible rectangle.
**              If num_rects is zero the surface must be completely contained
**              by the device.
**
** bounds:      The unclipped surface rectangle in device coords.  Extends
**              beyond the device bounds if the surface is not totally on
**              the device.
**
** rect[]:      An array of visible rectangles in device coords.  If num_rects
**              is non-zero only the region described by these rectangles is
**              copied to the frame buffer during a flush operation.
*/
typedef struct
{
        UInt32        num_rects;
        IOAccelBounds bounds;
        IOAccelBounds rect[0];
} IOAccelDeviceRegion;


/*
** Determine the size of a region.
*/
#define IOACCEL_SIZEOF_DEVICE_REGION(_rgn_) (sizeof(IOAccelDeviceRegion) + (_rgn_)->num_rects * sizeof(IOAccelBounds))


/*
** Surface client public memory types.  Private memory types start with
** kIOAccelNumSurfaceMemoryTypes.
*/
enum eIOAccelSurfaceMemoryTypes {
        kIOAccelNumSurfaceMemoryTypes
};


/*
** Surface client public methods.  Private methods start with
** kIOAccelNumSurfaceMethods.
*/
enum eIOAccelSurfaceMethods {
        kIOAccelSurfaceReadLockOptions,
        kIOAccelSurfaceReadUnlockOptions,
        kIOAccelSurfaceGetState,
        kIOAccelSurfaceWriteLockOptions,
        kIOAccelSurfaceWriteUnlockOptions,
        kIOAccelSurfaceRead,
        kIOAccelSurfaceSetShapeBacking,

        kIOAccelSurfaceSetIDMode,
        kIOAccelSurfaceSetScale,

        kIOAccelSurfaceSetShape,
        kIOAccelSurfaceFlush,

        kIOAccelSurfaceQueryLock,

        kIOAccelSurfaceReadLock,
        kIOAccelSurfaceReadUnlock,
        kIOAccelSurfaceWriteLock,
        kIOAccelSurfaceWriteUnlock,

        kIOAccelSurfaceControl,
        kIOAccelSurfaceSetShapeBackingAndLength,

        kIOAccelNumSurfaceMethods
};


/*
** Option bits for IOAccelCreateSurface and the kIOAccelSurfaceSetIDMode method.
** The color depth field can take any value of the _CGSDepth enumeration.
*/
typedef enum {
        kIOAccelSurfaceModeColorDepth1555  = 0x00000003,  
        kIOAccelSurfaceModeColorDepth8888  = 0x00000004,  
//      kIOAccelSurfaceModeColorDepthRGB565 = 0x00000005,  
        kIOAccelSurfaceModeColorDepthYUV   = 0x00000006,
        kIOAccelSurfaceModeColorDepthYUV9  = 0x00000007,
        kIOAccelSurfaceModeColorDepthYUV12 = 0x00000008,
        kIOAccelSurfaceModeColorDepthYUV2  = 0x00000009,
        kIOAccelSurfaceModeColorDepthBGRA32 = 0x0000000A,

//      kIOAccelSurfaceModeColorDepthRGBA64       = 0x0000000B,
//      kIOAccelSurfaceModeColorDepthRGBAFloat64  = 0x0000000C,
//      kIOAccelSurfaceModeColorDepthRGBAFloat128 = 0x0000000D,
        
//      kIOAccelSurfaceModeColorDepthYUV420  = 0x0000000E,
        kIOAccelSurfaceModeColorDepth2101010 = 0x0000000F,
        
        kIOAccelSurfaceModeColorDepthBits  = 0x0000000F,

        kIOAccelSurfaceModeStereoBit       = 0x00000010,
        kIOAccelSurfaceModeWindowedBit     = 0x00000020,

        kIOAccelSurfaceModeBeamSync        = 0x00008000
} eIOAccelSurfaceModeBits;


/*
** Options bits for IOAccelSetSurfaceShape and the kIOAccelSurfaceSetShape method.
*/
typedef enum {
        kIOAccelSurfaceShapeNone             = 0x00000000,
        kIOAccelSurfaceShapeNonBlockingBit   = 0x00000001,
        kIOAccelSurfaceShapeNonSimpleBit     = 0x00000002,
        kIOAccelSurfaceShapeIdentityScaleBit = 0x00000004,
        kIOAccelSurfaceShapeFrameSyncBit     = 0x00000008,
        kIOAccelSurfaceShapeBeamSyncBit      = 0x00000010,
        kIOAccelSurfaceShapeStaleBackingBit  = 0x00000020,
        kIOAccelSurfaceShapeAssemblyBit      = 0x00000040,
        kIOAccelSurfaceShapeWaitEnabledBit   = 0x00000080,

        /* wrong name, use kIOAccelSurfaceShapeNonBlockingBit */
        kIOAccelSurfaceShapeBlockingBit      = kIOAccelSurfaceShapeNonBlockingBit
} eIOAccelSurfaceShapeBits;

/*
** Return bits for the kIOAccelSurfaceGetState method.
*/
typedef enum {
        kIOAccelSurfaceStateNone    = 0x00000000,
        kIOAccelSurfaceStateIdleBit = 0x00000001
} eIOAccelSurfaceStateBits;

/*
** Option bits for the kIOAccelSurfaceSetScale method.
*/
typedef enum {
        kIOAccelSurfaceBeamSyncSwaps = 0x00000001,
        kIOAccelSurfaceFixedSource   = 0x00000002,

        kIOAccelSurfaceFiltering     = 0x000000f0,
        kIOAccelSurfaceFilterDefault = 0x00000000,
        kIOAccelSurfaceFilterNone    = 0x00000010,
        kIOAccelSurfaceFilterLinear  = 0x00000020

} eIOAccelSurfaceScaleBits;

/*
** Option bits for the kIOAccelSurfaceLock methods.
*/
typedef enum {
    kIOAccelSurfaceLockInBacking  = 0,
    kIOAccelSurfaceLockInAccel    = 1,
    kIOAccelSurfaceLockInDontCare = 2,
    kIOAccelSurfaceLockInMask     = 0x00000003
} eIOAccelSurfaceLockBits;

#endif /* _IOACCEL_SURFACE_CONNECT_H */

