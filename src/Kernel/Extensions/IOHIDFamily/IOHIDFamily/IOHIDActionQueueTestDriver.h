/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 22, 2022.
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

//
//  IOHIDActionQueueTestDriver.h
//  IOHIDFamily
//
//  Created by dekom on 3/14/18.
//

#ifndef IOHIDActionQueueTestDriver_h
#define IOHIDActionQueueTestDriver_h

#include <IOKit/hid/IOHIDActionQueue.h>
#include <IOKit/IOService.h>
#include <IOKit/IOWorkLoop.h>
#include <IOKit/IOInterruptEventSource.h>

class IOHIDActionQueueTestDriver: public IOService
{
    OSDeclareDefaultStructors(IOHIDActionQueueTestDriver)
private:
    IOWorkLoop              *_actionWL;
    IOHIDActionQueue        *_actionQueue;
    
    UInt32                  _actionCounter;
    
    IOWorkLoop              *_testWL1;
    IOInterruptEventSource  *_interrupt1;
    UInt32                  _actionCounter1;
    
    IOWorkLoop              *_testWL2;
    IOInterruptEventSource  *_interrupt2;
    UInt32                  _actionCounter2;
    
    void interruptAction1(IOInterruptEventSource *sender, int count);
    void interruptAction2(IOInterruptEventSource *sender, int count);
    
    void cancelHandlerCall();
    void runTest();
    
public:
    virtual bool start(IOService *provider) APPLE_KEXT_OVERRIDE;
    virtual void stop(IOService *provider) APPLE_KEXT_OVERRIDE;
    virtual void free() APPLE_KEXT_OVERRIDE;
    virtual IOReturn setProperties(OSObject *properties) APPLE_KEXT_OVERRIDE;
};



#endif /* IOHIDActionQueueTestDriver_h */
