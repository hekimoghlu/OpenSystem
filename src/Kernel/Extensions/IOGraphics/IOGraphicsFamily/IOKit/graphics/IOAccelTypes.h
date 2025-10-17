/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 14, 2025.
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
#ifndef _IOACCEL_TYPES_H
#define _IOACCEL_TYPES_H

#include <IOKit/IOTypes.h>
#include <IOKit/IOKitKeys.h>

#define IOACCEL_TYPES_REV       12

#if !defined(OSTYPES_K64_REV) && !defined(MAC_OS_X_VERSION_10_6)
#define IOACCELTYPES_10_5       1
#endif

/* Integer rectangle in device coordinates */
typedef struct
{
    SInt16      x;
    SInt16      y;
    SInt16      w;
    SInt16      h;
} IOAccelBounds;

typedef struct
{
    SInt16      w;
    SInt16      h;
} IOAccelSize;

/* Surface information */

enum {
    kIOAccelVolatileSurface     = 0x00000001
};

typedef struct
{
#if IOACCELTYPES_10_5
    vm_address_t        address[4];
#else
    mach_vm_address_t   address[4];
#endif /* IOACCELTYPES_10_5 */
    UInt32              rowBytes;
    UInt32              width;
    UInt32              height;
    UInt32              pixelFormat;
    IOOptionBits        flags;
    IOFixed             colorTemperature[4];
    UInt32              typeDependent[4];
} IOAccelSurfaceInformation;

typedef struct
{
#if IOACCELTYPES_10_5
        long x, y, w, h;
        void *client_addr;
        unsigned long client_row_bytes;
#else
        SInt32                    x, y, w, h;
        mach_vm_address_t client_addr;
        UInt32            client_row_bytes;
#endif /* IOACCELTYPES_10_5 */
} IOAccelSurfaceReadData;

typedef struct {
        IOAccelBounds   buffer;
        IOAccelSize     source;
        UInt32          reserved[8];
} IOAccelSurfaceScaling;


typedef SInt32 IOAccelID;

enum {
    kIOAccelPrivateID           = 0x00000001
};


#endif /* _IOACCEL_TYPES_H */

