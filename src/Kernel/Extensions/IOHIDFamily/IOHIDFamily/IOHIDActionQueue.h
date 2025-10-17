/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 14, 2025.
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
#ifndef IOHIDActionQueue_h
#define IOHIDActionQueue_h

#include <libkern/c++/OSContainers.h>
#include <IOKit/IOLocks.h>

class IOHIDAction;
class IOInterruptEventSource;
class IOWorkLoop;
class IOCommandGate;

class IOHIDActionQueue : public OSObject
{
    OSDeclareDefaultStructors(IOHIDActionQueue)
public:
    typedef void (^IOHIDActionBlock)();
    
private:
    OSObject                    *_owner;
    IOLock                      *_lock;
    OSArray                     *_actionArray;
    IOWorkLoop                  *_workLoop;
    IOCommandGate               *_commandGate;
    IOInterruptEventSource      *_actionInterrupt;
    UInt32                      _state __attribute__((aligned(sizeof(UInt32))));
    IOHIDActionBlock            _cancelHandler;
    
    void enqueueIOHIDAction(IOHIDAction *action);
    IOHIDAction *dequeueIOHIDAction();
    void handleIOHIDAction(IOInterruptEventSource *sender, int count);
    
protected:
    virtual void free() APPLE_KEXT_OVERRIDE;
    virtual bool init(OSObject *owner, IOWorkLoop *workLoop);
    
public:
    static IOHIDActionQueue *actionQueue(OSObject *owner, IOWorkLoop *workLoop);
    
    void activate();
    void cancel();
    void setCancelHandler(IOHIDActionBlock handler);
    
    void dispatchAsync(IOHIDActionBlock action);
    void dispatchSync(IOHIDActionBlock action);
};

#endif /* IOHIDActionQueue_h */
