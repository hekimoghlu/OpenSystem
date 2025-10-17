/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 20, 2024.
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
 *  BLDumpVolumeFinderInfo.c
 *  bless
 *
 *  Created by Shantonu Sen <ssen@apple.com> on Thu Apr 19 2001.
 *  Copyright (c) 2001-2007 Apple Inc. All Rights Reserved.
 *
 *  $Id: BLDumpVolumeFinderInfo.c,v 1.24 2006/02/20 22:49:54 ssen Exp $
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <sys/param.h>

#include <CoreFoundation/CoreFoundation.h>

#include "bless.h"
#include "bless_private.h"

/*
 * 1. getattrlist on the mountpoint to get the volume id
 * 2. read in the finder words
 * 3. for the directories we're interested in, get the entries in /.vol
 */
int BLCreateVolumeInformationDictionary(BLContextPtr context, const char * mountpoint,
					CFDictionaryRef *outDict) {
    uint32_t finderinfo[8];
    int err;
    uint32_t i;
    uint32_t dirID;
    CFMutableDictionaryRef dict = NULL;
    CFMutableArrayRef infarray = NULL;

    char blesspath[MAXPATHLEN];

    err = BLGetVolumeFinderInfo(context, mountpoint, finderinfo);
    if(err) {
        return 1;
    }

    infarray = CFArrayCreateMutable(kCFAllocatorDefault,
				    8,
				    &kCFTypeArrayCallBacks);

    dict =  CFDictionaryCreateMutable(kCFAllocatorDefault, 3, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);

    for(i = 0; i< 8-2; i++) {
      CFMutableDictionaryRef word =
	CFDictionaryCreateMutable(kCFAllocatorDefault,6, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
      CFTypeRef val;
      
      dirID = finderinfo[i];
      blesspath[0] = '\0';
      
      err = BLLookupFileIDOnMount(context, mountpoint, dirID, blesspath);
      
      val = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &dirID);
      CFDictionaryAddValue(word, CFSTR("Directory ID"), val);
      CFRelease(val); val = NULL;
      
      val = CFStringCreateWithCString(kCFAllocatorDefault, blesspath, kCFStringEncodingUTF8);
      CFDictionaryAddValue(word, CFSTR("Path"), val);
      CFRelease(val); val = NULL;

      if(strlen(blesspath) == 0 || 0 == strcmp(mountpoint, "/")) {
	  val = CFStringCreateWithCString(kCFAllocatorDefault, blesspath, kCFStringEncodingUTF8);
      } else {
	  val = CFStringCreateWithCString(kCFAllocatorDefault, blesspath+strlen(mountpoint), kCFStringEncodingUTF8);
      }
      CFDictionaryAddValue(word, CFSTR("Relative Path"), val);
      CFRelease(val); val = NULL;
      
      CFArrayAppendValue(infarray, word);
      CFRelease(word); word = NULL;
    }

    CFDictionaryAddValue(dict, CFSTR("Finder Info"),
			 infarray);

    CFRelease(infarray); infarray = NULL;

    {
        CFNumberRef vsdbref = NULL;
        uint64_t vsdb;
        vsdb = (*(uint64_t *)&finderinfo[8-2]);
        
        vsdbref = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &vsdb);
        CFDictionaryAddValue(dict, CFSTR("VSDB ID"), vsdbref);
        CFRelease(vsdbref); vsdbref = NULL;
    }
    
    *outDict = dict;
    return 0;
}
