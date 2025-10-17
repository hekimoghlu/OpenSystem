/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 29, 2023.
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
#ifndef _IOCDMEDIABSDCLIENT_H
#define _IOCDMEDIABSDCLIENT_H

#include <sys/ioctl.h>

#include <IOKit/storage/IOCDTypes.h>

/*
 * Definitions
 *
 * ioctl                        description
 * ---------------------------- ------------------------------------------------
 * DKIOCCDREAD                  see IOCDMedia::readCD()           in IOCDMedia.h
 * DKIOCCDREADTOC               see IOCDMedia::readTOC()          in IOCDMedia.h
 *
 * DKIOCCDREADDISCINFO          see IOCDMedia::readDiscInfo()     in IOCDMedia.h
 * DKIOCCDREADTRACKINFO         see IOCDMedia::readTrackInfo()    in IOCDMedia.h
 *
 * DKIOCCDREADISRC              see IOCDMedia::readISRC()         in IOCDMedia.h
 * DKIOCCDREADMCN               see IOCDMedia::readMCN()          in IOCDMedia.h
 *
 * DKIOCCDGETSPEED              see IOCDMedia::getSpeed()         in IOCDMedia.h
 * DKIOCCDSETSPEED              see IOCDMedia::setSpeed()         in IOCDMedia.h
 *
 *         in /System/Library/Frameworks/Kernel.framework/Headers/IOKit/storage/
 */

typedef struct
{
    uint64_t offset;

    uint8_t  sectorArea;
    uint8_t  sectorType;

#ifdef __LP64__
    uint8_t  reserved0080[10];                     /* reserved, clear to zero */
#else /* !__LP64__ */
    uint8_t  reserved0080[6];                      /* reserved, clear to zero */
#endif /* !__LP64__ */

    uint32_t bufferLength;                         /* actual length on return */
    void *   buffer;
} dk_cd_read_t;

typedef struct
{
    CDISRC   isrc;
    uint8_t  track;

    uint8_t  reserved0112[2];                      /* reserved, clear to zero */
} dk_cd_read_isrc_t;

typedef struct
{
    CDMCN    mcn;

    uint8_t  reserved0112[2];                      /* reserved, clear to zero */
} dk_cd_read_mcn_t;

typedef struct
{
    uint8_t  format;
    uint8_t  formatAsTime;

    uint8_t  reserved0016[5];                      /* reserved, clear to zero */

    union
    {
        uint8_t session;
        uint8_t track;
    } address;

#ifdef __LP64__
    uint8_t  reserved0064[6];                      /* reserved, clear to zero */
#else /* !__LP64__ */
    uint8_t  reserved0064[2];                      /* reserved, clear to zero */
#endif /* !__LP64__ */

    uint16_t bufferLength;                         /* actual length on return */
    void *   buffer;
} dk_cd_read_toc_t;

typedef struct
{
#ifdef __LP64__
    uint8_t  reserved0000[14];                     /* reserved, clear to zero */
#else /* !__LP64__ */
    uint8_t  reserved0000[10];                     /* reserved, clear to zero */
#endif /* !__LP64__ */

    uint16_t bufferLength;                         /* actual length on return */
    void *   buffer;
} dk_cd_read_disc_info_t;

typedef struct
{
    uint8_t  reserved0000[4];                      /* reserved, clear to zero */

    uint32_t address;
    uint8_t  addressType;

#ifdef __LP64__
    uint8_t  reserved0072[5];                      /* reserved, clear to zero */
#else /* !__LP64__ */
    uint8_t  reserved0072[1];                      /* reserved, clear to zero */
#endif /* !__LP64__ */

    uint16_t bufferLength;                         /* actual length on return */
    void *   buffer;
} dk_cd_read_track_info_t;

#define DKIOCCDREAD          _IOWR('d', 96, dk_cd_read_t)

#define DKIOCCDREADISRC      _IOWR('d', 97, dk_cd_read_isrc_t)
#define DKIOCCDREADMCN       _IOWR('d', 98, dk_cd_read_mcn_t)

#define DKIOCCDGETSPEED      _IOR('d', 99, uint16_t)
#define DKIOCCDSETSPEED      _IOW('d', 99, uint16_t)

#define DKIOCCDREADTOC       _IOWR('d', 100, dk_cd_read_toc_t)

#define DKIOCCDREADDISCINFO  _IOWR('d', 101, dk_cd_read_disc_info_t)
#define DKIOCCDREADTRACKINFO _IOWR('d', 102, dk_cd_read_track_info_t)

#ifdef KERNEL
#ifdef __cplusplus

/*
 * Kernel
 */

#include <IOKit/storage/IOCDMedia.h>
#include <IOKit/storage/IOMediaBSDClient.h>

/*
 * Class
 */

class IOCDMediaBSDClient : public IOMediaBSDClient
{
    OSDeclareDefaultStructors(IOCDMediaBSDClient)

protected:

    struct ExpansionData { /* */ };
    ExpansionData * _expansionData;

public:

    /*
     * Obtain this object's provider.  We override the superclass's method
     * to return a more specific subclass of IOService -- IOCDMedia.  This
     * method serves simply as a convenience to subclass developers.
     */

    virtual IOCDMedia * getProvider() const;

    /*
     * Process a CD-specific ioctl.
     */

    virtual int ioctl(dev_t dev, u_long cmd, caddr_t data, int flags, proc_t proc);

    OSMetaClassDeclareReservedUnused(IOCDMediaBSDClient, 0);
    OSMetaClassDeclareReservedUnused(IOCDMediaBSDClient, 1);
    OSMetaClassDeclareReservedUnused(IOCDMediaBSDClient, 2);
    OSMetaClassDeclareReservedUnused(IOCDMediaBSDClient, 3);
    OSMetaClassDeclareReservedUnused(IOCDMediaBSDClient, 4);
    OSMetaClassDeclareReservedUnused(IOCDMediaBSDClient, 5);
    OSMetaClassDeclareReservedUnused(IOCDMediaBSDClient, 6);
    OSMetaClassDeclareReservedUnused(IOCDMediaBSDClient, 7);
};

#endif /* __cplusplus */
#endif /* KERNEL */
#endif /* !_IOCDMEDIABSDCLIENT_H */
