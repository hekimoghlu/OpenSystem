/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 28, 2023.
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
#ifndef _IOBDMEDIABSDCLIENT_H
#define _IOBDMEDIABSDCLIENT_H

#include <sys/ioctl.h>
#include <sys/types.h>

#include <IOKit/storage/IOBDTypes.h>

/*
 * Definitions
 *
 * ioctl                        description
 * ---------------------------- ------------------------------------------------
 * DKIOCBDREADSTRUCTURE         see IOBDMedia::readStructure()    in IOBDMedia.h
 *
 * DKIOCBDREADDISCINFO          see IOBDMedia::readDiscInfo()     in IOBDMedia.h
 * DKIOCBDREADTRACKINFO         see IOBDMedia::readTrackInfo()    in IOBDMedia.h
 *
 * DKIOCBDREPORTKEY             see IOBDMedia::reportKey()        in IOBDMedia.h
 * DKIOCBDSENDKEY               see IOBDMedia::sendKey()          in IOBDMedia.h
 *
 * DKIOCBDGETSPEED              see IOBDMedia::getSpeed()         in IOBDMedia.h
 * DKIOCBDSETSPEED              see IOBDMedia::setSpeed()         in IOBDMedia.h
 *
 *         in /System/Library/Frameworks/Kernel.framework/Headers/IOKit/storage/
 */

__exported_push
typedef struct
{
    uint8_t  format;

    uint8_t  reserved0008[3];                      /* reserved, clear to zero */

    uint32_t address;
    uint8_t  grantID;
    uint8_t  layer;

    uint8_t  reserved0080[4];                      /* reserved, clear to zero */

    uint16_t bufferLength;
    void *   buffer;
} dk_bd_read_structure_t;

typedef struct
{
    uint8_t  format;
    uint8_t  keyClass;
    uint8_t  blockCount;

    uint8_t  reserved0024[1];                         /* reserved, clear to zero */

    uint32_t address;
    uint8_t  grantID;

    uint8_t  reserved0072[5];                      /* reserved, clear to zero */

    uint16_t bufferLength;
    void *   buffer;
} dk_bd_report_key_t;

typedef struct
{
    uint8_t  format;
    uint8_t  keyClass;

    uint8_t  reserved0016[6];                      /* reserved, clear to zero */

    uint8_t  grantID;

    uint8_t  reserved0072[5];                      /* reserved, clear to zero */

    uint16_t bufferLength;
    void *   buffer;
} dk_bd_send_key_t;

typedef struct
{
    uint8_t  reserved0000[14];                     /* reserved, clear to zero */

    uint16_t bufferLength;                         /* actual length on return */
    void *   buffer;
} dk_bd_read_disc_info_t;

typedef struct
{
    uint8_t  reserved0000[4];                      /* reserved, clear to zero */

    uint32_t address;
    uint8_t  addressType;

    uint8_t  reserved0072[5];                      /* reserved, clear to zero */

    uint16_t bufferLength;                         /* actual length on return */
    void *   buffer;
} dk_bd_read_track_info_t;
__exported_pop

#define DKIOCBDREADSTRUCTURE   _IOW('d', 160, dk_bd_read_structure_t)
#define DKIOCBDREPORTKEY       _IOW('d', 161, dk_bd_report_key_t)
#define DKIOCBDSENDKEY         _IOW('d', 162, dk_bd_send_key_t)

#define DKIOCBDGETSPEED        _IOR('d', 163, uint16_t)
#define DKIOCBDSETSPEED        _IOW('d', 163, uint16_t)

#define DKIOCBDREADDISCINFO    _IOWR('d', 164, dk_bd_read_disc_info_t)
#define DKIOCBDREADTRACKINFO   _IOWR('d', 165, dk_bd_read_track_info_t)

#define DKIOCBDSPLITTRACK      _IOW('d', 166, uint32_t)

#ifdef KERNEL
#ifdef __cplusplus

/*
 * Kernel
 */

#include <IOKit/storage/IOBDMedia.h>
#include <IOKit/storage/IOMediaBSDClient.h>

/*
 * Class
 */

class __exported IOBDMediaBSDClient : public IOMediaBSDClient
{
    OSDeclareDefaultStructors(IOBDMediaBSDClient)

protected:

    struct ExpansionData { /* */ };
    ExpansionData * _expansionData;

public:

    /*
     * Obtain this object's provider.  We override the superclass's method
     * to return a more specific subclass of IOService -- IOBDMedia.  This
     * method serves simply as a convenience to subclass developers.
     */

    virtual IOBDMedia * getProvider() const;

    /*
     * Process a BD-specific ioctl.
     */

    virtual int ioctl(dev_t dev, u_long cmd, caddr_t data, int flags, proc_t proc);

    OSMetaClassDeclareReservedUnused(IOBDMediaBSDClient, 0);
    OSMetaClassDeclareReservedUnused(IOBDMediaBSDClient, 1);
    OSMetaClassDeclareReservedUnused(IOBDMediaBSDClient, 2);
    OSMetaClassDeclareReservedUnused(IOBDMediaBSDClient, 3);
    OSMetaClassDeclareReservedUnused(IOBDMediaBSDClient, 4);
    OSMetaClassDeclareReservedUnused(IOBDMediaBSDClient, 5);
    OSMetaClassDeclareReservedUnused(IOBDMediaBSDClient, 6);
    OSMetaClassDeclareReservedUnused(IOBDMediaBSDClient, 7);
};

#endif /* __cplusplus */
#endif /* KERNEL */
#endif /* !_IOBDMEDIABSDCLIENT_H */
