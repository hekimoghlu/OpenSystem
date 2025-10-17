/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 8, 2022.
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
 *  BLCreateEFIXMLRepresentationForNetworkPath.c
 *  bless
 *
 *  Created by Shantonu Sen on 11/15/05.
 *  Copyright 2005-2007 Apple Inc. All Rights Reserved.
 *
 */

#import <IOKit/IOKitLib.h>
#import <IOKit/IOCFSerialize.h>
#import <IOKit/IOBSD.h>
#import <IOKit/IOKitKeys.h>
#include <IOKit/network/IONetworkInterface.h>
#include <IOKit/network/IONetworkController.h>

#import <CoreFoundation/CoreFoundation.h>

#include <string.h>
#include <sys/param.h>
#include <sys/stat.h>

#include "bless.h"
#include "bless_private.h"

int BLCreateEFIXMLRepresentationForNetworkPath(BLContextPtr context,
                                               BLNetBootProtocolType protocol,
                                               const char *interface,
                                               const char *host,
                                               const char *path,
                                               const char *optionalData,
                                               CFStringRef *xmlString)
{
    mach_port_t masterPort;
    kern_return_t kret;
    io_service_t iface;
    
    CFDataRef xmlData;
    CFMutableDictionaryRef dict, matchDict;
    CFMutableArrayRef array;
    CFDataRef macAddress;
    
    const UInt8 *xmlBuffer;
    UInt8 *outBuffer;
    CFIndex count;
    
    kret = IOMasterPort(MACH_PORT_NULL, &masterPort);
    if(kret) return 1;
        
    
    
    array = CFArrayCreateMutable(kCFAllocatorDefault, 0, &kCFTypeArrayCallBacks);
    
    dict = CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks,
                                     &kCFTypeDictionaryValueCallBacks);
    
    matchDict = IOBSDNameMatching(masterPort, 0, interface);
    CFDictionarySetValue(matchDict, CFSTR(kIOProviderClassKey), CFSTR(kIONetworkInterfaceClass));

    CFRetain(matchDict);
    iface = IOServiceGetMatchingService(masterPort,
                                        matchDict);
    
    if(iface == IO_OBJECT_NULL) {
        contextprintf(context, kBLLogLevelError, "Could not find object for %s\n", interface);
        CFRelease(matchDict);
        CFRelease(dict);
        CFRelease(array);
        return 1;
    }
    
    CFDictionaryAddValue(dict, CFSTR("IOMatch"), matchDict);
    CFRelease(matchDict);
    
    macAddress = IORegistryEntrySearchCFProperty(iface, kIOServicePlane,
                                                 CFSTR(kIOMACAddress),
                                                 kCFAllocatorDefault,
                                                 kIORegistryIterateRecursively|kIORegistryIterateParents);
    if(macAddress) {
        contextprintf(context, kBLLogLevelVerbose, "MAC address %s found for %s\n",
					  BLGetCStringDescription(macAddress), interface);
        
        CFDictionaryAddValue(dict, CFSTR("BLMACAddress"), macAddress);
        CFRelease(macAddress);
    } else {
        contextprintf(context, kBLLogLevelVerbose, "No MAC address found for %s\n", interface);        
    }
    
    IOObjectRelease(iface);
    
    CFArrayAppendValue(array, dict);
    CFRelease(dict);
    
    if(host) {
        CFStringRef hostString;
        
        hostString = CFStringCreateWithCString(kCFAllocatorDefault, host, kCFStringEncodingUTF8);

        dict = CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks,
                                         &kCFTypeDictionaryValueCallBacks);
        CFDictionaryAddValue(dict, CFSTR("IOEFIDevicePathType"),
                             CFSTR("MessagingIPv4"));
        CFDictionaryAddValue(dict, CFSTR("RemoteIpAddress"),
                             hostString);
        CFArrayAppendValue(array, dict);
        CFRelease(dict);
        CFRelease(hostString);
        
        if(path) {
            CFStringRef pathString;
            
            pathString = CFStringCreateWithCString(kCFAllocatorDefault, path, kCFStringEncodingUTF8);
            
            dict = CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks,
                                             &kCFTypeDictionaryValueCallBacks);
            CFDictionaryAddValue(dict, CFSTR("IOEFIDevicePathType"),
                                 CFSTR("MediaFilePath"));
            CFDictionaryAddValue(dict, CFSTR("Path"),
                                 pathString);
            CFArrayAppendValue(array, dict);
            CFRelease(dict);            
            CFRelease(pathString);
        }
        
    }

    contextprintf(context, kBLLogLevelVerbose, "Netboot protocol %d\n", protocol);        
    if (protocol == kBLNetBootProtocol_PXE) {
        dict = CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks,
                                         &kCFTypeDictionaryValueCallBacks);
        CFDictionaryAddValue(dict, CFSTR("IOEFIDevicePathType"),
                             CFSTR("MessagingNetbootProtocol"));
        CFDictionaryAddValue(dict, CFSTR("Protocol"),
                             CFSTR("FE3913DB-9AEE-4E40-A294-ABBE93A1A4B7"));
        CFArrayAppendValue(array, dict);
        CFRelease(dict);
    }
    
    if(optionalData) {
        CFStringRef optString = CFStringCreateWithCString(kCFAllocatorDefault, optionalData, kCFStringEncodingUTF8);
        
        dict = CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks,
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

