/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 12, 2024.
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
#include <IOKit/storage/IOBDBlockStorageDriver.h>
#include <IOKit/storage/IOBDMedia.h>

#define super IODVDBlockStorageDriver
OSDefineMetaClassAndStructors(IOBDBlockStorageDriver, IODVDBlockStorageDriver)

OSCompileAssert(sizeof(BDDiscInfo) == sizeof(CDDiscInfo));
OSCompileAssert(sizeof(BDTrackInfo) == sizeof(CDTrackInfo));

#define reportDiscInfo(x)    reportDiscInfo((CDDiscInfo *)(x))
#define reportTrackInfo(y,x) reportTrackInfo((y),(CDTrackInfo *)(x))

IOBDBlockStorageDevice *
IOBDBlockStorageDriver::getProvider() const
{
    return (IOBDBlockStorageDevice *) IOService::getProvider();
}

/* Accept a new piece of media, doing whatever's necessary to make it
 * show up properly to the system.
 */
IOReturn
IOBDBlockStorageDriver::acceptNewMedia(void)
{
    if (getMediaType() < kBDMediaTypeMin || getMediaType() > kBDMediaTypeMax) {
        return super::acceptNewMedia();
    }

    /* Obtain disc status: */

    switch (getMediaType()) {
        case kBDMediaTypeR: {
            BDDiscInfo discInfo;
            BDTrackInfo trackInfo;
            IOReturn result;

            result = reportDiscInfo(&discInfo);
            if (result != kIOReturnSuccess) {
                break;
            }

            /* Obtain track status: */

            if (discInfo.discStatus == 0x01) { /* is disc incomplete? */
                UInt16 trackLast = (discInfo.lastTrackNumberInLastSessionMSB << 8) |
                                    discInfo.lastTrackNumberInLastSessionLSB;

                _writeProtected = false;

                result = reportTrackInfo(trackLast,&trackInfo);
                if (result != kIOReturnSuccess) {
                    break;
                }

                _maxBlockNumber = max( _maxBlockNumber,
                                       max( OSSwapBigToHostInt32(trackInfo.trackStartAddress) +
                                            OSSwapBigToHostInt32(trackInfo.trackSize), 1 ) - 1 );
            } else if (discInfo.discStatus == 0x03) { /* is disc random recordable? */
                _writeProtected = false;
            }

            break;
        }
    }

    return IOBlockStorageDriver::acceptNewMedia();
}

const char *
IOBDBlockStorageDriver::getDeviceTypeName(void)
{
    return(kIOBlockStorageDeviceTypeBD);
}

IOMedia *
IOBDBlockStorageDriver::instantiateDesiredMediaObject(void)
{
    if (getMediaType() < kBDMediaTypeMin || getMediaType() > kBDMediaTypeMax) {
        return super::instantiateDesiredMediaObject();
    }

    return(new IOBDMedia);
}

IOMedia *
IOBDBlockStorageDriver::instantiateMediaObject(UInt64 base,UInt64 byteSize,
                                        UInt32 blockSize,char *mediaName)
{
    IOMedia *media = NULL;

    if (getMediaType() < kBDMediaTypeMin || getMediaType() > kBDMediaTypeMax) {
        return super::instantiateMediaObject(base,byteSize,blockSize,mediaName);
    }

    media = IOBlockStorageDriver::instantiateMediaObject(
                                             base,byteSize,blockSize,mediaName);

    if (media) {
        const char *description = NULL;
        const char *picture = NULL;

        switch (getMediaType()) {
            case kBDMediaTypeROM:
                description = kIOBDMediaTypeROM;
                picture = "BD.icns";
                break;
            case kBDMediaTypeR:
                description = kIOBDMediaTypeR;
                picture = "BD-R.icns";
                break;
            case kBDMediaTypeRE:
                description = kIOBDMediaTypeRE;
                picture = "BD-RE.icns";
                break;
        }

        if (description) {
            media->setProperty(kIOBDMediaTypeKey, description);
        }

        if (picture) {
            OSDictionary *dictionary = OSDictionary::withCapacity(2);
            OSString *identifier = OSString::withCString("com.apple.iokit.IOBDStorageFamily");
            OSString *resourceFile = OSString::withCString(picture);

            if (dictionary && identifier && resourceFile) {
                dictionary->setObject("CFBundleIdentifier", identifier);
                dictionary->setObject("IOBundleResourceFile", resourceFile);
            }

            media->setProperty(kIOMediaIconKey, dictionary);

            if (resourceFile) {
                resourceFile->release();
            }
            if (identifier) {
                identifier->release();
            }
            if (dictionary) {
                dictionary->release();
            }
        }
    }

    return media;
}

IOReturn
IOBDBlockStorageDriver::readStructure(IOMemoryDescriptor *buffer,const DVDStructureFormat format,
                                        const UInt32 address,const UInt8 layer,const UInt8 agid)
{
    if (getMediaType() < kBDMediaTypeMin || getMediaType() > kBDMediaTypeMax) {
        return super::readStructure(buffer,format,address,layer,agid);
    }

    return(getProvider()->readDiscStructure(buffer,format,address,layer,agid,1));
}

IOReturn
IOBDBlockStorageDriver::splitTrack(UInt32 address)
{
    return(getProvider()->splitTrack(address));
}

OSMetaClassDefineReservedUnused(IOBDBlockStorageDriver,  0);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDriver,  1);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDriver,  2);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDriver,  3);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDriver,  4);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDriver,  5);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDriver,  6);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDriver,  7);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDriver,  8);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDriver,  9);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDriver, 10);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDriver, 11);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDriver, 12);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDriver, 13);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDriver, 14);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDriver, 15);
