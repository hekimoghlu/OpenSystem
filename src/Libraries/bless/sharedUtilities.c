/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 3, 2025.
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
#include <stdio.h>
#include "sharedUtilities.h"

CFDataRef CreateDataFromFileURL(CFAllocatorRef alloc, CFURLRef fileURL, CFErrorRef *error)
{
    CFDataRef result = NULL;
    CFNumberRef fileSizeValue;
    // get the file size from the file URL
    if (CFURLCopyResourcePropertyForKey(fileURL, kCFURLFileSizeKey, &fileSizeValue, error)) {
        if (fileSizeValue) {
            CFIndex fileSize;
            // get the fileSize as a CFIndex
            if (CFNumberGetValue(fileSizeValue, kCFNumberCFIndexType, &fileSize)) {
                if (fileSize == 0) {
                    // zero-length file, return a zero-length CFData
                    result = CFDataCreate(alloc, NULL, 0);
                } else {
                    CFAllocatorRef bytesAllocator;
                    // we need a non-NULL allocator to use with CFDataCreateWithBytesNoCopy
                    if (alloc != NULL) {
                        bytesAllocator = alloc;
                    } else {
                        bytesAllocator = kCFAllocatorSystemDefault;
                    }
                    // create the read stream
                    CFReadStreamRef readStream = CFReadStreamCreateWithFile(kCFAllocatorDefault, fileURL);
                    if (readStream) {
                        // open the read stream
                        if (CFReadStreamOpen(readStream)) {
                            // allocate the mutableBytes buffer to read into
                            UInt8 *mutableBytes = CFAllocatorAllocate(bytesAllocator, fileSize, 0);
                            if (mutableBytes) {
                                // read the file into the mutableBytes
                                CFIndex lengthRead = CFReadStreamRead(readStream, mutableBytes, fileSize);
                                if (lengthRead >= 0) {
                                    // create a CFData with mutableBytes
                                    result = CFDataCreateWithBytesNoCopy(bytesAllocator, mutableBytes, lengthRead, bytesAllocator);
                                    if (!result && error) {
                                        *error = CFErrorCreate(alloc, kCFErrorDomainPOSIX, ENOMEM, NULL);
                                    }
                                    // else success!
                                } else if (error) {
                                    *error = CFReadStreamCopyError(readStream);
                                }
                                if (!result) {
                                    CFAllocatorDeallocate(bytesAllocator, mutableBytes);
                                }
                                // else mutableBytes will be released when result is released
                            } else if (error) {
                                *error = CFErrorCreate(alloc, kCFErrorDomainPOSIX, ENOMEM, NULL);
                            }
                            CFReadStreamClose(readStream);
                        } else if (error) {
                            *error = CFReadStreamCopyError(readStream);
                        }
                        CFRelease(readStream);
                    } else if (error) {
                        // this should never happen unless a non-file URL is passed
                        *error = CFErrorCreate(alloc, kCFErrorDomainOSStatus, kIOReturnBadArgument, NULL);
                    }
                }
            } else if (error) {
                // the file size was larger than a CFIndex
                *error = CFErrorCreate(alloc, kCFErrorDomainPOSIX, EFBIG, NULL);
            }
        } else if (error) {
            // this should never happen unless a non-file URL is passed
            *error = CFErrorCreate(alloc, kCFErrorDomainOSStatus, kIOReturnBadArgument, NULL);
        }
    }
    return result;
}
