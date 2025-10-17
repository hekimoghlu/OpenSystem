/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 25, 2024.
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
/*
 * This header contains the IOAppleLabelScheme class definition.
 */

#ifndef _IOAPPLELABELSCHEME_H
#define _IOAPPLELABELSCHEME_H

#include <IOKit/IOTypes.h>

/*
 * kIOAppleLabelSchemeClass is the name of the IOAppleLabelScheme class.
 */

#define kIOAppleLabelSchemeClass "IOAppleLabelScheme"

/*
 * Apple Label Scheme Definitions
 */

#pragma pack(push, 1)                        /* (enable 8-bit struct packing) */

/* Label scheme. */

struct applelabel
{
    uint8_t  al_boot0[416];               /* (reserved for boot area)         */
    uint16_t al_magic;                    /* (the magic number)               */
    uint16_t al_type;                     /* (label type)                     */
    uint32_t al_flags;                    /* (generic flags)                  */
    uint64_t al_offset;                   /* (offset of property area, bytes) */
    uint32_t al_size;                     /* (size of property area, bytes)   */
    uint32_t al_checksum;                 /* (checksum of property area)      */
    uint8_t  al_boot1[72];                /* (reserved for boot area)         */
};

/* Label scheme signature (al_magic). */

#define AL_MAGIC 0x414C

/* Label scheme version (al_type). */

#define AL_TYPE_DEFAULT 0x0000

/* Label scheme flags (al_flags). */

#define AL_FLAG_DEFAULT 0x00000000

#pragma pack(pop)                        /* (reset to default struct packing) */

#ifdef KERNEL
#ifdef __cplusplus

/*
 * Kernel
 */

#include <IOKit/storage/IOFilterScheme.h>

/*
 * Class
 */

class __exported IOAppleLabelScheme : public IOFilterScheme
{
    OSDeclareDefaultStructors(IOAppleLabelScheme);

protected:

    struct ExpansionData { /* */ };
    ExpansionData * _expansionData;

    IOMedia * _content;

    /*
     * Free all of this object's outstanding resources.
     */

    virtual void free(void) APPLE_KEXT_OVERRIDE;

    /*
     * Scan the provider media for an Apple label scheme.
     */

    virtual IOMedia * scan(SInt32 * score);

    /*
     * Ask whether the given content appears to be corrupt.
     */

    virtual bool isContentCorrupt(OSDictionary * properties);

    /*
     * Ask whether the given content appears to be invalid.
     */

    virtual bool isContentInvalid(OSDictionary * properties);

    /*
     * Instantiate a new media object to represent the given content.
     */

    virtual IOMedia * instantiateMediaObject(OSDictionary * properties);

    /*
     * Allocate a new media object (called from instantiateMediaObject).
     */

    virtual IOMedia * instantiateDesiredMediaObject(OSDictionary * properties);

    /*
     * Attach the given media object to the device tree plane.
     */

    virtual bool attachMediaObjectToDeviceTree(IOMedia * media);

    /*
     * Detach the given media object from the device tree plane.
     */

    virtual void detachMediaObjectFromDeviceTree(IOMedia * media);

public:

    /*
     * Initialize this object's minimal state.
     */

    virtual bool init(OSDictionary * properties = 0) APPLE_KEXT_OVERRIDE;

    /*
     * Determine whether the provider media contains an Apple label scheme.
     */

    virtual IOService * probe(IOService * provider, SInt32 * score) APPLE_KEXT_OVERRIDE;

    /*
     * Publish the new media object which represents our content.
     */

    virtual bool start(IOService * provider) APPLE_KEXT_OVERRIDE;

    /*
     * Clean up after the media object we published before terminating.
     */

    virtual void stop(IOService * provider) APPLE_KEXT_OVERRIDE;

    OSMetaClassDeclareReservedUnused(IOAppleLabelScheme,  0);
    OSMetaClassDeclareReservedUnused(IOAppleLabelScheme,  1);
    OSMetaClassDeclareReservedUnused(IOAppleLabelScheme,  2);
    OSMetaClassDeclareReservedUnused(IOAppleLabelScheme,  3);
    OSMetaClassDeclareReservedUnused(IOAppleLabelScheme,  4);
    OSMetaClassDeclareReservedUnused(IOAppleLabelScheme,  5);
    OSMetaClassDeclareReservedUnused(IOAppleLabelScheme,  6);
    OSMetaClassDeclareReservedUnused(IOAppleLabelScheme,  7);
    OSMetaClassDeclareReservedUnused(IOAppleLabelScheme,  8);
    OSMetaClassDeclareReservedUnused(IOAppleLabelScheme,  9);
    OSMetaClassDeclareReservedUnused(IOAppleLabelScheme, 10);
    OSMetaClassDeclareReservedUnused(IOAppleLabelScheme, 11);
    OSMetaClassDeclareReservedUnused(IOAppleLabelScheme, 12);
    OSMetaClassDeclareReservedUnused(IOAppleLabelScheme, 13);
    OSMetaClassDeclareReservedUnused(IOAppleLabelScheme, 14);
    OSMetaClassDeclareReservedUnused(IOAppleLabelScheme, 15);
};

#endif /* __cplusplus */
#endif /* KERNEL */
#endif /* !_IOAPPLELABELSCHEME_H */
