/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 5, 2025.
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
 *  BLCreateEFIXMLRepresentationForDevice.c
 *  bless
 *
 *  Created by Shantonu Sen on 12/2/05.
 *  Copyright 2005-2007 Apple Inc. All Rights Reserved.
 *
 */
#include <TargetConditionals.h>

#import <IOKit/IOKitLib.h>
#import <IOKit/IOCFSerialize.h>
#import <IOKit/IOBSD.h>
#import <IOKit/IOKitKeys.h>
#import <IOKit/storage/IOMedia.h>

#import <CoreFoundation/CoreFoundation.h>

#include <string.h>
#include <sys/param.h>
#include <sys/stat.h>

#include "bless.h"
#include "bless_private.h"

extern int addMatchingInfoForBSDName(BLContextPtr context,
                              mach_port_t masterPort,
                              CFMutableDictionaryRef dict,
                              const char *bsdName,
                              bool shortForm);

int BLCreateEFIXMLRepresentationForDevice(BLContextPtr context,
                                        const char *bsdName,
                                        const char *optionalData,
                                        CFStringRef *xmlString,
                                        bool shortForm)
{
    int ret;
    mach_port_t masterPort;
    kern_return_t kret;
    
    CFDataRef xmlData;
    CFMutableDictionaryRef dict;
    CFMutableArrayRef array;
    
    const UInt8 *xmlBuffer;
    UInt8 *outBuffer;
    CFIndex count;
    
    kret = IOMasterPort(MACH_PORT_NULL, &masterPort);
    if(kret) return 1;
    
    array = CFArrayCreateMutable(kCFAllocatorDefault, 0, &kCFTypeArrayCallBacks);
    
    dict = CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks,
                                     &kCFTypeDictionaryValueCallBacks);
    
    ret = addMatchingInfoForBSDName(context, masterPort, dict, bsdName, shortForm);
    if(ret) {
        CFRelease(dict);
        CFRelease(array);
        return 2;
    }    
    CFArrayAppendValue(array, dict);
    CFRelease(dict);
    
    if(optionalData) {
        CFStringRef optString = CFStringCreateWithCString(kCFAllocatorDefault, optionalData, kCFStringEncodingUTF8);
        
        dict = CFDictionaryCreateMutable(kCFAllocatorDefault, 0,
                                         &kCFTypeDictionaryKeyCallBacks,
                                         &kCFTypeDictionaryValueCallBacks);
        CFDictionaryAddValue(dict, CFSTR("IOEFIBootOption"),
                             optString);
        CFArrayAppendValue(array, dict);
        CFRelease(dict);        
        
        CFRelease(optString);
    }
    
    xmlData = IOCFSerialize(array, 0);
    CFRelease(array);
    
    if(xmlData == NULL) {
        contextprintf(context, kBLLogLevelError, "Can't create XML representation\n");
        return 2;
    }
    
    count = CFDataGetLength(xmlData);
    xmlBuffer = CFDataGetBytePtr(xmlData);
    outBuffer = calloc(count+1, sizeof(char)); // terminate
    
    memcpy(outBuffer, xmlBuffer, count);
    CFRelease(xmlData);
    
    *xmlString = CFStringCreateWithCString(kCFAllocatorDefault, (const char *)outBuffer, kCFStringEncodingUTF8);
    
    free(outBuffer);
    
    return 0;
}

