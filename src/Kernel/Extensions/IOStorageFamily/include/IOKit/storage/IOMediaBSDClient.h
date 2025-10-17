/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 12, 2024.
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
#ifndef _IOMEDIABSDCLIENT_H
#define _IOMEDIABSDCLIENT_H

#include <sys/disk.h>

#ifdef KERNEL
#ifdef __cplusplus

/*
 * Kernel
 */

#include <sys/conf.h>
#include <IOKit/storage/IOMedia.h>

class  AnchorTable;
class  MinorTable;
struct MinorSlot;

__exported UInt64 _IOMediaBSDClientGetThrottleMask(IOMedia * media);

/*
 * Class
 */

class __exported IOMediaBSDClient : public IOService
{
    OSDeclareDefaultStructors(IOMediaBSDClient)

protected:

    struct ExpansionData { /* */ };
    ExpansionData * _expansionData;

private:

    AnchorTable * _anchors;
    UInt32        _reserved0064 __attribute__ ((unused));
    UInt32        _reserved0096 __attribute__ ((unused));
    MinorTable *  _minors;
    UInt32        _reserved0160 __attribute__ ((unused));

protected:

    /*
     * Find the whole media that roots the given media tree.
     */

    virtual IOMedia * getWholeMedia( IOMedia * media,
                                     UInt32 *  slicePathSize = 0,
                                     char *    slicePath     = 0 );

    /*
     * Create bdevsw and cdevsw nodes for the given media object.
     */

    virtual bool createNodes(IOMedia * media);

    /*
     * Free all of this object's outstanding resources.
     */

    virtual void free() APPLE_KEXT_OVERRIDE;

public:

    /*
     * Obtain this object's provider.  We override the superclass's method to
     * return a more specific subclass of IOService -- IOMedia.   This method
     * method serves simply as a convenience to subclass developers.
     */

    virtual IOMedia * getProvider() const APPLE_KEXT_OVERRIDE;

    /*
     * Initialize this object's minimal state.
     */

    virtual bool init(OSDictionary * properties  = 0) APPLE_KEXT_OVERRIDE;

    /*
     * This method is called once we have been attached to the provider object.
     */

    virtual bool start(IOService * provider) APPLE_KEXT_OVERRIDE;

    /*
     * This method is called when we are to terminate from the provider object.
     */

    virtual bool terminate(IOOptionBits options) APPLE_KEXT_OVERRIDE;

    /*
     * Process a foreign ioctl.
     */

    virtual int ioctl(dev_t dev, u_long cmd, caddr_t data, int flags, proc_t proc);

    OSMetaClassDeclareReservedUnused(IOMediaBSDClient,  0);
    OSMetaClassDeclareReservedUnused(IOMediaBSDClient,  1);
    OSMetaClassDeclareReservedUnused(IOMediaBSDClient,  2);
    OSMetaClassDeclareReservedUnused(IOMediaBSDClient,  3);
    OSMetaClassDeclareReservedUnused(IOMediaBSDClient,  4);
    OSMetaClassDeclareReservedUnused(IOMediaBSDClient,  5);
    OSMetaClassDeclareReservedUnused(IOMediaBSDClient,  6);
    OSMetaClassDeclareReservedUnused(IOMediaBSDClient,  7);
    OSMetaClassDeclareReservedUnused(IOMediaBSDClient,  8);
    OSMetaClassDeclareReservedUnused(IOMediaBSDClient,  9);
    OSMetaClassDeclareReservedUnused(IOMediaBSDClient, 10);
    OSMetaClassDeclareReservedUnused(IOMediaBSDClient, 11);
    OSMetaClassDeclareReservedUnused(IOMediaBSDClient, 12);
    OSMetaClassDeclareReservedUnused(IOMediaBSDClient, 13);
    OSMetaClassDeclareReservedUnused(IOMediaBSDClient, 14);
    OSMetaClassDeclareReservedUnused(IOMediaBSDClient, 15);
};

#endif /* __cplusplus */
#endif /* KERNEL */
#endif /* !_IOMEDIABSDCLIENT_H */
