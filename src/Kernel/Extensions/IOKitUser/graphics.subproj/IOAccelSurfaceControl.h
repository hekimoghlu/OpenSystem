/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 10, 2022.
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
#ifndef _IOACCEL_SURFACE_CONTROL_H
#define _IOACCEL_SURFACE_CONTROL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <IOKit/graphics/IOAccelSurfaceConnect.h>

#define IOACCEL_SURFACE_CONTROL_REV     8

typedef struct IOAccelConnectStruct *IOAccelConnect;


/* Create an accelerated surface and attach it to a CGS surface */
IOReturn IOAccelCreateSurface( io_service_t service, UInt32 wid, eIOAccelSurfaceModeBits modebits, IOAccelConnect *connect );
 
/* Fix surface size & scaling */
IOReturn IOAccelSetSurfaceScale( IOAccelConnect connect, IOOptionBits options,
                                    IOAccelSurfaceScaling * scaling, UInt32 scalingSize );

/* Detach an an accelerated surface from a CGS surface and destroy it*/
IOReturn IOAccelDestroySurface( IOAccelConnect connect );

/* Change the visible region of the accelerated surface */
IOReturn IOAccelSetSurfaceFramebufferShapeWithBacking( IOAccelConnect connect, IOAccelDeviceRegion *rgn,
                                            eIOAccelSurfaceShapeBits options, UInt32 framebufferIndex,
                                            IOVirtualAddress backing, UInt32 rowbytes );

IOReturn IOAccelSetSurfaceFramebufferShapeWithBackingAndLength( IOAccelConnect connect, IOAccelDeviceRegion *rgn,
                                            eIOAccelSurfaceShapeBits options, UInt32 framebufferIndex,
                                            IOVirtualAddress backing, UInt32 rowbytes, UInt32 backingLength );

IOReturn IOAccelSetSurfaceFramebufferShape( IOAccelConnect connect, IOAccelDeviceRegion *rgn,
                                            eIOAccelSurfaceShapeBits options, UInt32 framebufferIndex );

/* Block until the last visible region change applied to an accelerated surface is complete */
IOReturn IOAccelWaitForSurface( IOAccelConnect connect );

/* Get the back buffer of the surface.  Supplies client virtual address. */

IOReturn IOAccelWriteLockSurfaceWithOptions( IOAccelConnect connect, IOOptionBits options,
                                             IOAccelSurfaceInformation * info, UInt32 infoSize );
IOReturn IOAccelWriteUnlockSurfaceWithOptions( IOAccelConnect connect, IOOptionBits options );
IOReturn IOAccelReadLockSurfaceWithOptions( IOAccelConnect connect, IOOptionBits options,
                                            IOAccelSurfaceInformation * info, UInt32 infoSize );
IOReturn IOAccelReadUnlockSurfaceWithOptions( IOAccelConnect connect, IOOptionBits options );

IOReturn IOAccelQueryLockSurface( IOAccelConnect connect );
IOReturn IOAccelWriteLockSurface( IOAccelConnect connect, IOAccelSurfaceInformation * info, UInt32 infoSize );
IOReturn IOAccelWriteUnlockSurface( IOAccelConnect connect );
IOReturn IOAccelReadLockSurface( IOAccelConnect connect, IOAccelSurfaceInformation * info, UInt32 infoSize );
IOReturn IOAccelReadUnlockSurface( IOAccelConnect connect );

/* Flush surface to visible region */
IOReturn IOAccelFlushSurfaceOnFramebuffers( IOAccelConnect connect, IOOptionBits options, UInt32 framebufferMask );


/* Read surface back buffer */
IOReturn IOAccelReadSurface( IOAccelConnect connect, IOAccelSurfaceReadData * parameters );

IOReturn IOAccelCreateAccelID(IOOptionBits options, IOAccelID * identifier);
IOReturn IOAccelDestroyAccelID(IOOptionBits options, IOAccelID identifier);

IOReturn IOAccelSurfaceControl( IOAccelConnect connect,
                                    UInt32 selector, UInt32 arg, UInt32 * result);

#ifdef __cplusplus
}
#endif

#endif /* _IOACCEL_SURFACE_CONTROL_H */

