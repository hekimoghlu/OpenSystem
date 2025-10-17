/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 22, 2023.
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
#ifndef _IOKIT_IOBOOTFRAMEBUFFER_H
#define _IOKIT_IOBOOTFRAMEBUFFER_H

#include <IOKit/IOPlatformExpert.h>

#include <IOKit/graphics/IOFramebuffer.h>


class IOBootFramebuffer : public IOFramebuffer
{
    OSDeclareDefaultStructors(IOBootFramebuffer)

public:

    virtual IOService * probe(  IOService *     provider,
                                SInt32 *        score ) APPLE_KEXT_OVERRIDE;

//    virtual bool start( IOService * provider ) APPLE_KEXT_OVERRIDE;

    virtual const char * getPixelFormats( void ) APPLE_KEXT_OVERRIDE;

    virtual IOItemCount getDisplayModeCount( void ) APPLE_KEXT_OVERRIDE;

    virtual IOReturn getDisplayModes( IODisplayModeID * allDisplayModes ) APPLE_KEXT_OVERRIDE;

    virtual IOReturn getInformationForDisplayMode( IODisplayModeID displayMode,
                    IODisplayModeInformation * info ) APPLE_KEXT_OVERRIDE;

    virtual UInt64  getPixelFormatsForDisplayMode( IODisplayModeID displayMode,
                    IOIndex depth ) APPLE_KEXT_OVERRIDE;

    virtual IOReturn getPixelInformation(
        IODisplayModeID displayMode, IOIndex depth,
        IOPixelAperture aperture, IOPixelInformation * pixelInfo ) APPLE_KEXT_OVERRIDE;

    virtual IOReturn getCurrentDisplayMode( IODisplayModeID * displayMode,
                            IOIndex * depth ) APPLE_KEXT_OVERRIDE;

    virtual IODeviceMemory * getApertureRange( IOPixelAperture aperture ) APPLE_KEXT_OVERRIDE;

    virtual bool isConsoleDevice( void ) APPLE_KEXT_OVERRIDE;

    virtual IOReturn setCLUTWithEntries( IOColorEntry * colors, UInt32 index,
                UInt32 numEntries, IOOptionBits options ) APPLE_KEXT_OVERRIDE;

    virtual IOReturn setGammaTable( UInt32 channelCount, UInt32 dataCount,
                    UInt32 dataWidth, void * data ) APPLE_KEXT_OVERRIDE;
};

#endif /* ! _IOKIT_IOBOOTFRAMEBUFFER_H */

