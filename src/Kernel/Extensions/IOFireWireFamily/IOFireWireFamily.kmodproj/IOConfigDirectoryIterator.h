/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 3, 2024.
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
// system
#import <libkern/c++/OSIterator.h>
#import <libkern/c++/OSSet.h>
#import <IOKit/IOLib.h>

class IOConfigDirectory;

class IOConfigDirectoryIterator : public OSIterator
{
    OSDeclareDefaultStructors(IOConfigDirectoryIterator)

protected:
    OSSet *	fDirectorySet;
    OSIterator * fDirectoryIterator;
	
    virtual void free(void) APPLE_KEXT_OVERRIDE;

public:
    virtual IOReturn init(IOConfigDirectory *owner, UInt32 testVal, UInt32 testMask);
    
    virtual void reset(void) APPLE_KEXT_OVERRIDE;

    virtual bool isValid(void) APPLE_KEXT_OVERRIDE;

    virtual OSObject *getNextObject(void) APPLE_KEXT_OVERRIDE;
};
