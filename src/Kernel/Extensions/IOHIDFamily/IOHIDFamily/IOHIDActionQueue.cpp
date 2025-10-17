/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 14, 2025.
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
#include "IOHIDActionQueue.h"
#include <IOKit/IOInterruptEventSource.h>
#include <IOKit/IOWorkLoop.h>
#include <IOKit/IOCommandGate.h>
#include <AssertMacros.h>
#include <libkern/Block.h>
#include <IOKit/assert.h>

typedef enum {
    kIOHIDActionQueueStateInactive      = 0,
    kIOHIDActionQueueStateActivated     = 1 << 0,
    kIOHIDActionQueueStateCancelled     = 1 << 1
} IOHIDActionQueueState;

class IOHIDAction : public OSObject
{
    OSDeclareDefaultStructors(IOHIDAction)
    
    IOHIDActionQueue::IOHIDActionBlock _action;
protected:
    virtual void free() APPLE_KEXT_OVERRIDE {
        if (_action) {
            Block_release(_action);
        }
        OSObject::free();
    }
    
public:
    static IOHIDAction *hidAction(IOHIDActionQueue::IOHIDActionBlock action) {
        if (!action) {
            return 0;
        }
        
        IOHIDAction *me = new IOHIDAction;
        
        if (me && !me->init()) {
            me->release();
            return 0;
        }
        
        me->_action = Block_copy(action);
        
        return me;
    }
    
    virtual void runAction() {
        (_action)();
    }
};

OSDefineMetaClassAndStructors(IOHIDAction, OSObject)

#define super OSObject

OSDefineMetaClassAndStructors(IOHIDActionQueue, OSObject)

bool IOHIDActionQueue::init(OSObject *owner, IOWorkLoop *workLoop)
{
    bool result = false;
    
    require(owner && workLoop, exit);
    require(super::init(), exit);
    
    _owner = owner;
    _workLoop = workLoop;
    _workLoop->retain();
    
    _commandGate = IOCommandGate::commandGate(this);
    require(_commandGate, exit);
    
    _actionArray = OSArray::withCapacity(1);
    require(_actionArray, exit);
    
    _lock = IOLockAlloc();
    require(_lock, exit);
    
    _actionInterrupt = IOInterruptEventSource::interruptEventSource(this,
                                                                    OSMemberFunctionCast(IOInterruptEventAction,
                                                                                         this, &IOHIDActionQueue::handleIOHIDAction));
    require(_actionInterrupt, exit);
    
    result = true;
    
exit:
    return result;
}

void IOHIDActionQueue::free()
{
    if (_state) {
        assert(_state & kIOHIDActionQueueStateCancelled);
    }
    
    OSSafeReleaseNULL(_workLoop);
    OSSafeReleaseNULL(_commandGate);
    OSSafeReleaseNULL(_actionArray);
    OSSafeReleaseNULL(_actionInterrupt);
    
    if (_lock) {
        IOLockFree(_lock);
        _lock = NULL;
    }
    
    super::free();
}

IOHIDActionQueue *IOHIDActionQueue::actionQueue(OSObject *owner, IOWorkLoop *workLoop)
{
    IOHIDActionQueue *me = new IOHIDActionQueue;
    
    if (me && !me->init(owner, workLoop)) {
        me->release();
        return 0;
    }
    
    return me;
}

void IOHIDActionQueue::enqueueIOHIDAction(IOHIDAction *action)
{
    unsigned int count = 0;
    
    IOLockLock(_lock);
    _actionArray->setObject(action);
    count = _actionArray->getCount();
    IOLockUnlock(_lock);
    
    if (count == 1) {
        _actionInterrupt->interruptOccurred(NULL, NULL, 0);
    }
}

IOHIDAction *IOHIDActionQueue::dequeueIOHIDAction()
{
    IOHIDAction *action = NULL;
    
    IOLockLock(_lock);
    if (_actionArray->getCount()) {
        action = (IOHIDAction *)_actionArray->getObject(0);
        if (action) {
            action->retain();
            _actionArray->removeObject(0);
        }
    }
    IOLockUnlock(_lock);
    
    return action;
}

void IOHIDActionQueue::handleIOHIDAction(IOInterruptEventSource *sender __unused, int count __unused)
{
    IOHIDAction *action = NULL;
    
    while ((action = dequeueIOHIDAction())) {
        action->runAction();
        action->release();
    }
}

void IOHIDActionQueue::dispatchAsync(IOHIDActionBlock action)
{
    IOHIDAction *hidAction = NULL;
    
    assert(_state == kIOHIDActionQueueStateActivated);
    
    hidAction = IOHIDAction::hidAction(action);
    if (hidAction) {
        enqueueIOHIDAction(hidAction);
        hidAction->release();
    }
}

void IOHIDActionQueue::dispatchSync(IOHIDActionBlock action)
{
    assert(_state == kIOHIDActionQueueStateActivated);
    
    _commandGate->runActionBlock(^IOReturn{
        (action)();
        return kIOReturnSuccess;
    });
}

void IOHIDActionQueue::activate()
{
    if (OSBitOrAtomic(kIOHIDActionQueueStateActivated, &_state) & kIOHIDActionQueueStateActivated) {
        return;
    }
    
    _workLoop->addEventSource(_actionInterrupt);
    _workLoop->addEventSource(_commandGate);
}

void IOHIDActionQueue::cancel()
{
    IOHIDAction *hidAction = NULL;
    
    if (OSBitOrAtomic(kIOHIDActionQueueStateCancelled, &_state) & kIOHIDActionQueueStateCancelled) {
        return;
    }
    
    IOHIDActionBlock cancelBlock = ^() {
        _workLoop->removeEventSource(_actionInterrupt);
        _workLoop->removeEventSource(_commandGate);
        
        if (_cancelHandler) {
            (_cancelHandler)();
            Block_release(_cancelHandler);
        }
    };
    
    hidAction = IOHIDAction::hidAction(cancelBlock);
    if (hidAction) {
        enqueueIOHIDAction(hidAction);
        hidAction->release();
    }
}

void IOHIDActionQueue::setCancelHandler(IOHIDActionBlock handler)
{
    assert(_state == kIOHIDActionQueueStateInactive);
    
    if (handler) {
        _cancelHandler = Block_copy(handler);
    }
}
