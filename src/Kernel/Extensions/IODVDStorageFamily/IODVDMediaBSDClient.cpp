/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 25, 2025.
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
#include <sys/errno.h>
#include <sys/proc.h>
#include <IOKit/storage/IODVDMediaBSDClient.h>

#define super IOMediaBSDClient
OSDefineMetaClassAndStructors(IODVDMediaBSDClient, IOMediaBSDClient)

typedef struct
{
    uint8_t       format;

    uint8_t       reserved0008[3];

    uint32_t      address;
    uint8_t       grantID;
    uint8_t       layer;

    uint16_t      bufferLength;
    user32_addr_t buffer;
} dk_dvd_read_structure_32_t;

typedef struct
{
    uint8_t       format;

    uint8_t       reserved0008[3];

    uint32_t      address;
    uint8_t       grantID;
    uint8_t       layer;

    uint8_t       reserved0080[4];

    uint16_t      bufferLength;
    user64_addr_t buffer;
} dk_dvd_read_structure_64_t;

typedef struct
{
    uint8_t       format;
    uint8_t       keyClass;
    uint8_t       blockCount;

    uint8_t       reserved0024[1];

    uint32_t      address;
    uint8_t       grantID;

    uint8_t       reserved0072[1];

    uint16_t      bufferLength;
    user32_addr_t buffer;
} dk_dvd_report_key_32_t;

typedef struct
{
    uint8_t       format;
    uint8_t       keyClass;
    uint8_t       blockCount;

    uint8_t       reserved0024[1];

    uint32_t      address;
    uint8_t       grantID;

    uint8_t       reserved0072[5];

    uint16_t      bufferLength;
    user64_addr_t buffer;
} dk_dvd_report_key_64_t;

typedef struct
{
    uint8_t       format;
    uint8_t       keyClass;

    uint8_t       reserved0016[6];

    uint8_t       grantID;

    uint8_t       reserved0072[1];

    uint16_t      bufferLength;
    user32_addr_t buffer;
} dk_dvd_send_key_32_t;

typedef struct
{
    uint8_t       format;
    uint8_t       keyClass;

    uint8_t       reserved0016[6];

    uint8_t       grantID;

    uint8_t       reserved0072[5];

    uint16_t      bufferLength;
    user64_addr_t buffer;
} dk_dvd_send_key_64_t;

typedef struct
{
    uint8_t       reserved0000[10];

    uint16_t      bufferLength;
    user32_addr_t buffer;
} dk_dvd_read_disc_info_32_t;

typedef struct
{
    uint8_t       reserved0000[14];

    uint16_t      bufferLength;
    user64_addr_t buffer;
} dk_dvd_read_disc_info_64_t;

typedef struct
{
    uint8_t       reserved0000[4];

    uint32_t      address;
    uint8_t       addressType;

    uint8_t       reserved0072[1];

    uint16_t      bufferLength;
    user32_addr_t buffer;
} dk_dvd_read_rzone_info_32_t;

typedef struct
{
    uint8_t       reserved0000[4];

    uint32_t      address;
    uint8_t       addressType;

    uint8_t       reserved0072[5];

    uint16_t      bufferLength;
    user64_addr_t buffer;
} dk_dvd_read_rzone_info_64_t;

#define DKIOCDVDREADSTRUCTURE32 _IOW('d', 128, dk_dvd_read_structure_32_t)
#define DKIOCDVDREADSTRUCTURE64 _IOW('d', 128, dk_dvd_read_structure_64_t)
#define DKIOCDVDREPORTKEY32     _IOW('d', 129, dk_dvd_report_key_32_t)
#define DKIOCDVDREPORTKEY64     _IOW('d', 129, dk_dvd_report_key_64_t)
#define DKIOCDVDSENDKEY32       _IOW('d', 130, dk_dvd_send_key_32_t)
#define DKIOCDVDSENDKEY64       _IOW('d', 130, dk_dvd_send_key_64_t)

#define DKIOCDVDREADDISCINFO32  _IOWR('d', 132, dk_dvd_read_disc_info_32_t)
#define DKIOCDVDREADDISCINFO64  _IOWR('d', 132, dk_dvd_read_disc_info_64_t)
#define DKIOCDVDREADRZONEINFO32 _IOWR('d', 133, dk_dvd_read_rzone_info_32_t)
#define DKIOCDVDREADRZONEINFO64 _IOWR('d', 133, dk_dvd_read_rzone_info_64_t)

static bool DKIOC_IS_RESERVED(caddr_t data, uint32_t reserved)
{
    UInt32 index;

    for ( index = 0; index < sizeof(reserved) * 8; index++, reserved >>= 1 )
    {
        if ( (reserved & 1) )
        {
            if ( data[index] )  return true;
        }
    }

    return false;
}

static IOMemoryDescriptor * DKIOC_PREPARE_BUFFER( user_addr_t address,
                                                  UInt32      length,
                                                  IODirection direction,
                                                  proc_t      proc )
{
    IOMemoryDescriptor * buffer = 0;

    if ( address && length )
    {
        buffer = IOMemoryDescriptor::withAddressRange(    // (create the buffer)
            /* address */ address,
            /* length  */ length,
            /* options */ direction,
            /* task    */ (proc == kernproc) ? kernel_task : current_task() );
    }

    if ( buffer )
    {
        if ( buffer->prepare() != kIOReturnSuccess )     // (prepare the buffer)
        {
            buffer->release();
            buffer = 0;
        }
    }

    return buffer;
}

static void DKIOC_COMPLETE_BUFFER(IOMemoryDescriptor * buffer)
{
    if ( buffer )
    {
        buffer->complete();                             // (complete the buffer)
        buffer->release();                               // (release the buffer)
    }
}

IODVDMedia * IODVDMediaBSDClient::getProvider() const
{
    //
    // Obtain this object's provider.   We override the superclass's method
    // to return a more specific subclass of IOService -- IODVDMedia.  This
    // method serves simply as a convenience to subclass developers.
    //

    return (IODVDMedia *) IOService::getProvider();
}

int IODVDMediaBSDClient::ioctl( dev_t   dev,
                                u_long  cmd,
                                caddr_t data,
                                int     flags,
                                proc_t  proc )
{
    //
    // Process a DVD-specific ioctl.
    //

    IOMemoryDescriptor * buffer = 0;
    int                  error  = 0;
    IOReturn             status = kIOReturnSuccess;

    switch ( cmd )
    {
        case DKIOCDVDREADSTRUCTURE32:          // (dk_dvd_read_structure_32_t *)
        {
            dk_dvd_read_structure_32_t * request;

            request = (dk_dvd_read_structure_32_t *) data;

            if ( proc_is64bit(proc) )  { error = ENOTTY;  break; }

            if ( DKIOC_IS_RESERVED(data, 0xE) )  { error = EINVAL;  break; }

            buffer = DKIOC_PREPARE_BUFFER(
                       /* address   */ request->buffer,
                       /* length    */ request->bufferLength,
                       /* direction */ kIODirectionIn,
                       /* proc      */ proc );

            status = getProvider()->readStructure(
                       /* buffer    */                      buffer,
                       /* format    */ (DVDStructureFormat) request->format,
                       /* address   */                      request->address,
                       /* layer     */                      request->layer,
                       /* grantID   */                      request->grantID );

            status = (status == kIOReturnUnderrun) ? kIOReturnSuccess : status;

            DKIOC_COMPLETE_BUFFER(buffer);

        } break;

        case DKIOCDVDREADSTRUCTURE64:          // (dk_dvd_read_structure_64_t *)
        {
            dk_dvd_read_structure_64_t * request;

            request = (dk_dvd_read_structure_64_t *) data;

            if ( proc_is64bit(proc) == 0 )  { error = ENOTTY;  break; }

            if ( DKIOC_IS_RESERVED(data, 0x3C0E) )  { error = EINVAL;  break; }

            buffer = DKIOC_PREPARE_BUFFER(
                       /* address   */ request->buffer,
                       /* length    */ request->bufferLength,
                       /* direction */ kIODirectionIn,
                       /* proc      */ proc );

            status = getProvider()->readStructure(
                       /* buffer    */                      buffer,
                       /* format    */ (DVDStructureFormat) request->format,
                       /* address   */                      request->address,
                       /* layer     */                      request->layer,
                       /* grantID   */                      request->grantID );

            status = (status == kIOReturnUnderrun) ? kIOReturnSuccess : status;

            DKIOC_COMPLETE_BUFFER(buffer);

        } break;

        case DKIOCDVDREPORTKEY32:                  // (dk_dvd_report_key_32_t *)
        {
            dk_dvd_report_key_32_t * request = (dk_dvd_report_key_32_t *) data;

            if ( proc_is64bit(proc) )  { error = ENOTTY;  break; }

            if ( DKIOC_IS_RESERVED(data, 0x208) )  { error = EINVAL;  break; }

            buffer = DKIOC_PREPARE_BUFFER(
                       /* address   */ request->buffer,
                       /* length    */ request->bufferLength,
                       /* direction */ kIODirectionIn,
                       /* proc      */ proc );

            status = getProvider()->reportKey(
                       /* buffer     */                buffer,
                       /* keyClass   */ (DVDKeyClass)  request->keyClass,
                       /* address    */                request->address,
                       /* blockCount */                request->blockCount,
                       /* grantID    */                request->grantID,
                       /* format     */ (DVDKeyFormat) request->format );

            status = (status == kIOReturnUnderrun) ? kIOReturnSuccess : status;

            DKIOC_COMPLETE_BUFFER(buffer);

        } break;

        case DKIOCDVDREPORTKEY64:                  // (dk_dvd_report_key_64_t *)
        {
            dk_dvd_report_key_64_t * request = (dk_dvd_report_key_64_t *) data;

            if ( proc_is64bit(proc) == 0 )  { error = ENOTTY;  break; }

            if ( DKIOC_IS_RESERVED(data, 0x3E08) )  { error = EINVAL;  break; }

            buffer = DKIOC_PREPARE_BUFFER(
                       /* address   */ request->buffer,
                       /* length    */ request->bufferLength,
                       /* direction */ kIODirectionIn,
                       /* proc      */ proc );

            status = getProvider()->reportKey(
                       /* buffer     */                buffer,
                       /* keyClass   */ (DVDKeyClass)  request->keyClass,
                       /* address    */                request->address,
                       /* blockCount */                request->blockCount,
                       /* grantID    */                request->grantID,
                       /* format     */ (DVDKeyFormat) request->format );

            status = (status == kIOReturnUnderrun) ? kIOReturnSuccess : status;

            DKIOC_COMPLETE_BUFFER(buffer);

        } break;

        case DKIOCDVDSENDKEY32:                      // (dk_dvd_send_key_32_t *)
        {
            dk_dvd_send_key_32_t * request = (dk_dvd_send_key_32_t *) data;

            if ( proc_is64bit(proc) )  { error = ENOTTY;  break; }

            if ( DKIOC_IS_RESERVED(data, 0x2FC) )  { error = EINVAL;  break; }

            buffer = DKIOC_PREPARE_BUFFER(
                       /* address   */ request->buffer,
                       /* length    */ request->bufferLength,
                       /* direction */ kIODirectionOut,
                       /* proc      */ proc );

            status = getProvider()->sendKey(
                       /* buffer    */                buffer,
                       /* keyClass  */ (DVDKeyClass)  request->keyClass,
                       /* grantID   */                request->grantID,
                       /* format    */ (DVDKeyFormat) request->format );

            status = (status == kIOReturnUnderrun) ? kIOReturnSuccess : status;

            DKIOC_COMPLETE_BUFFER(buffer);

        } break;

        case DKIOCDVDSENDKEY64:                      // (dk_dvd_send_key_64_t *)
        {
            dk_dvd_send_key_64_t * request = (dk_dvd_send_key_64_t *) data;

            if ( proc_is64bit(proc) == 0 )  { error = ENOTTY;  break; }

            if ( DKIOC_IS_RESERVED(data, 0x3EFC) )  { error = EINVAL;  break; }

            buffer = DKIOC_PREPARE_BUFFER(
                       /* address   */ request->buffer,
                       /* length    */ request->bufferLength,
                       /* direction */ kIODirectionOut,
                       /* proc      */ proc );

            status = getProvider()->sendKey(
                       /* buffer    */                buffer,
                       /* keyClass  */ (DVDKeyClass)  request->keyClass,
                       /* grantID   */                request->grantID,
                       /* format    */ (DVDKeyFormat) request->format );

            status = (status == kIOReturnUnderrun) ? kIOReturnSuccess : status;

            DKIOC_COMPLETE_BUFFER(buffer);

        } break;

        case DKIOCDVDGETSPEED:                                   // (uint16_t *)
        {
            status = getProvider()->getSpeed((uint16_t *)data);

        } break;

        case DKIOCDVDSETSPEED:                                   // (uint16_t *)
        {
            status = getProvider()->setSpeed(*(uint16_t *)data);

        } break;

        case DKIOCDVDREADDISCINFO32:           // (dk_dvd_read_disc_info_32_t *)
        {
            dk_dvd_read_disc_info_32_t * request;

            request = (dk_dvd_read_disc_info_32_t *) data;

            if ( proc_is64bit(proc) )  { error = ENOTTY;  break; }

            if ( DKIOC_IS_RESERVED(data, 0x3FF) )  { error = EINVAL;  break; }

            buffer = DKIOC_PREPARE_BUFFER(
                       /* address   */ request->buffer,
                       /* length    */ request->bufferLength,
                       /* direction */ kIODirectionIn,
                       /* proc      */ proc );

            status = getProvider()->readDiscInfo(
                       /* buffer          */ buffer,
                       /* actualByteCount */ &request->bufferLength );

            status = (status == kIOReturnUnderrun) ? kIOReturnSuccess : status;

            DKIOC_COMPLETE_BUFFER(buffer);

        } break;

        case DKIOCDVDREADDISCINFO64:           // (dk_dvd_read_disc_info_64_t *)
        {
            dk_dvd_read_disc_info_64_t * request;

            request = (dk_dvd_read_disc_info_64_t *) data;

            if ( proc_is64bit(proc) == 0 )  { error = ENOTTY;  break; }

            if ( DKIOC_IS_RESERVED(data, 0x3FFF) )  { error = EINVAL;  break; }

            buffer = DKIOC_PREPARE_BUFFER(
                       /* address   */ request->buffer,
                       /* length    */ request->bufferLength,
                       /* direction */ kIODirectionIn,
                       /* proc      */ proc );

            status = getProvider()->readDiscInfo(
                       /* buffer          */ buffer,
                       /* actualByteCount */ &request->bufferLength );

            status = (status == kIOReturnUnderrun) ? kIOReturnSuccess : status;

            DKIOC_COMPLETE_BUFFER(buffer);

        } break;

        case DKIOCDVDREADRZONEINFO32:         // (dk_dvd_read_rzone_info_32_t *)
        {
            dk_dvd_read_rzone_info_32_t * request;

            request = (dk_dvd_read_rzone_info_32_t *) data;

            if ( proc_is64bit(proc) )  { error = ENOTTY;  break; }

            if ( DKIOC_IS_RESERVED(data, 0x20F) )  { error = EINVAL;  break; }

            buffer = DKIOC_PREPARE_BUFFER(
                       /* address   */ request->buffer,
                       /* length    */ request->bufferLength,
                       /* direction */ kIODirectionIn,
                       /* proc      */ proc );

            status = getProvider()->readRZoneInfo(
                       /* buffer          */ buffer,
                       /* address         */ request->address,
                       /* addressType     */ request->addressType,
                       /* actualByteCount */ &request->bufferLength );

            status = (status == kIOReturnUnderrun) ? kIOReturnSuccess : status;

            DKIOC_COMPLETE_BUFFER(buffer);

        } break;

        case DKIOCDVDREADRZONEINFO64:         // (dk_dvd_read_rzone_info_64_t *)
        {
            dk_dvd_read_rzone_info_64_t * request;

            request = (dk_dvd_read_rzone_info_64_t *) data;

            if ( proc_is64bit(proc) == 0 )  { error = ENOTTY;  break; }

            if ( DKIOC_IS_RESERVED(data, 0x3E0F) )  { error = EINVAL;  break; }

            buffer = DKIOC_PREPARE_BUFFER(
                       /* address   */ request->buffer,
                       /* length    */ request->bufferLength,
                       /* direction */ kIODirectionIn,
                       /* proc      */ proc );

            status = getProvider()->readRZoneInfo(
                       /* buffer          */ buffer,
                       /* address         */ request->address,
                       /* addressType     */ request->addressType,
                       /* actualByteCount */ &request->bufferLength );

            status = (status == kIOReturnUnderrun) ? kIOReturnSuccess : status;

            DKIOC_COMPLETE_BUFFER(buffer);

        } break;

        default:
        {
            //
            // A foreign ioctl was received.  Ask our superclass' opinion.
            //

            error = super::ioctl(dev, cmd, data, flags, proc);

        } break;
    }

    return error ? error : getProvider()->errnoFromReturn(status);
}

OSMetaClassDefineReservedUnused(IODVDMediaBSDClient, 0);
OSMetaClassDefineReservedUnused(IODVDMediaBSDClient, 1);
OSMetaClassDefineReservedUnused(IODVDMediaBSDClient, 2);
OSMetaClassDefineReservedUnused(IODVDMediaBSDClient, 3);
OSMetaClassDefineReservedUnused(IODVDMediaBSDClient, 4);
OSMetaClassDefineReservedUnused(IODVDMediaBSDClient, 5);
OSMetaClassDefineReservedUnused(IODVDMediaBSDClient, 6);
OSMetaClassDefineReservedUnused(IODVDMediaBSDClient, 7);
