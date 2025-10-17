/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 19, 2025.
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
#include <pthread.h>
#include <CoreFoundation/CFRuntime.h>
#include <IOKit/hid/IOHIDDevicePlugIn.h>
#include "IOHIDLibPrivate.h"
#include "IOHIDDevice.h"
#include "IOHIDTransaction.h"

typedef struct {
    void              * context;
    IOHIDCallback       callback;
    IOHIDTransactionRef transaction;
} IOHIDTransactionCallbackInfo;

static IOHIDTransactionRef  __IOHIDTransactionCreate(
                                    CFAllocatorRef          allocator, 
                                    CFAllocatorContext *    context __unused);
static void                 __IOHIDTransactionExtRelease( CFTypeRef object );
static void                 __IOHIDTransactionIntRelease( CFTypeRef object );
static void                 __IOHIDTransactionCommitCallback(
                                    void *                  context,
                                    IOReturn                result,
                                    void *                  sender);


typedef struct __IOHIDTransaction
{
    IOHIDObjectBase                         hidBase;

    IOHIDDeviceTransactionInterface**       transactionInterface;
    
    CFTypeRef                               asyncEventSource;
    CFRunLoopRef                            runLoop;
    CFStringRef                             runLoopMode;

    IOHIDDeviceRef                          device;
    IOOptionBits                            options;
} __IOHIDTransaction, *__IOHIDTransactionRef;

static const IOHIDObjectClass __IOHIDTransactionClass = {
    {
        _kCFRuntimeCustomRefCount,      // version
        "IOHIDTransaction",             // className
        NULL,                           // init
        NULL,                           // copy
        __IOHIDTransactionExtRelease,   // finalize
        NULL,                           // equal
        NULL,                           // hash
        NULL,                           // copyFormattingDesc
        NULL,                           // copyDebugDesc
        NULL,                           // reclaim
        _IOHIDObjectExtRetainCount      // refcount
    },
    _IOHIDObjectIntRetainCount,
    __IOHIDTransactionIntRelease
};

static pthread_once_t   __transactionTypeInit = PTHREAD_ONCE_INIT;
static CFTypeID         __kIOHIDTransactionTypeID = _kCFRuntimeNotATypeID;

//------------------------------------------------------------------------------
// __IOHIDTransactionRegister
//------------------------------------------------------------------------------
void __IOHIDTransactionRegister(void)
{
    __kIOHIDTransactionTypeID = 
                _CFRuntimeRegisterClass(&__IOHIDTransactionClass.cfClass);
}

//------------------------------------------------------------------------------
// __IOHIDTransactionCreate
//------------------------------------------------------------------------------
IOHIDTransactionRef __IOHIDTransactionCreate(   
                                CFAllocatorRef              allocator, 
                                CFAllocatorContext *        context __unused)
{
    uint32_t    size;
    
    /* allocate service */
    size  = sizeof(__IOHIDTransaction) - sizeof(CFRuntimeBase);
    
    return (IOHIDTransactionRef)_IOHIDObjectCreateInstance(allocator, IOHIDTransactionGetTypeID(), size, NULL);;
}

//------------------------------------------------------------------------------
// __IOHIDTransactionExtRelease
//------------------------------------------------------------------------------
void __IOHIDTransactionExtRelease( CFTypeRef object __unused )
{
    
}

//------------------------------------------------------------------------------
// __IOHIDTransactionIntRelease
//------------------------------------------------------------------------------
void __IOHIDTransactionIntRelease( CFTypeRef object )
{
    IOHIDTransactionRef transaction = (IOHIDTransactionRef)object;
    
    if ( transaction->transactionInterface ) {
        (*transaction->transactionInterface)->Release(transaction->transactionInterface);
        transaction->transactionInterface = NULL;
    }
    
    if ((transaction->options & kIOHIDTransactionOptionsWeakDevice) == 0 && transaction->device ) {
        CFRelease(transaction->device);
        transaction->device = NULL;
    }
}

//------------------------------------------------------------------------------
// __IOHIDTransactionCommitCallback
//------------------------------------------------------------------------------
void __IOHIDTransactionCommitCallback(
                                    void *                  context,
                                    IOReturn                result,
                                    void *                  sender)
{
    IOHIDTransactionCallbackInfo * info = (IOHIDTransactionCallbackInfo *)context;

    if (info) {
        if ((info->transaction->transactionInterface == sender) && info->callback) {
            info->callback(info->context, result, info->transaction);
        }

        free(info);
    }
}

//------------------------------------------------------------------------------
// IOHIDTransactionGetTypeID
//------------------------------------------------------------------------------
CFTypeID IOHIDTransactionGetTypeID(void) 
{
    if ( _kCFRuntimeNotATypeID == __kIOHIDTransactionTypeID )
        pthread_once(&__transactionTypeInit, __IOHIDTransactionRegister);
        
    return __kIOHIDTransactionTypeID;
}

//------------------------------------------------------------------------------
// IOHIDTransactionCreate
//------------------------------------------------------------------------------
IOHIDTransactionRef IOHIDTransactionCreate(
                                CFAllocatorRef                  allocator, 
                                IOHIDDeviceRef                  device,
                                IOHIDTransactionDirectionType   direction,
                                IOOptionBits                    options)
{
    IOCFPlugInInterface **              deviceInterface         = NULL;
    IOHIDDeviceTransactionInterface **  transactionInterface    = NULL;
    IOHIDTransactionRef                 transaction             = NULL;
    IOReturn                            ret;
    
    if ( !device )
        return NULL;
        
    deviceInterface = _IOHIDDeviceGetIOCFPlugInInterface(device);
    
    if ( !deviceInterface )
        return NULL;
        
    ret = (*deviceInterface)->QueryInterface(
                        deviceInterface, 
                        CFUUIDGetUUIDBytes(kIOHIDDeviceTransactionInterfaceID), 
                        (LPVOID)&transactionInterface);
    
    if ( ret != kIOReturnSuccess || !transactionInterface )
        return NULL;
        
    transaction = __IOHIDTransactionCreate(allocator, NULL);
    
    if ( !transaction ) {
        (*transactionInterface)->Release(transactionInterface);
        return NULL;
    }

    transaction->transactionInterface   = transactionInterface;
    
    transaction->options = options;
    
    if (options & kIOHIDTransactionOptionsWeakDevice) {
        transaction->device = device;
    } else {
        transaction->device = (IOHIDDeviceRef)CFRetain(device);
    }
    
    (*transaction->transactionInterface)->setDirection(
                            transaction->transactionInterface, 
                            direction, 
                            options);
    
    return transaction;
}

//------------------------------------------------------------------------------
// IOHIDTransactionGetDirection
//------------------------------------------------------------------------------
IOHIDDeviceRef IOHIDTransactionGetDevice(     
                                IOHIDTransactionRef             transaction)
{
    return transaction->device;
}

//------------------------------------------------------------------------------
// IOHIDTransactionGetDirection
//------------------------------------------------------------------------------
IOHIDTransactionDirectionType IOHIDTransactionGetDirection(     
                                IOHIDTransactionRef             transaction)
{
    IOHIDTransactionDirectionType direction = 0;
    (*transaction->transactionInterface)->getDirection(
                                        transaction->transactionInterface, 
                                        &direction);
    
    return direction;
}

//------------------------------------------------------------------------------
// IOHIDTransactionSetDirection
//------------------------------------------------------------------------------
void IOHIDTransactionSetDirection(        
                                IOHIDTransactionRef             transaction,
                                IOHIDTransactionDirectionType   direction)
{
    (*transaction->transactionInterface)->setDirection(
                                        transaction->transactionInterface, 
                                        direction, 
                                        0);
}
                                
//------------------------------------------------------------------------------
// IOHIDTransactionAddElement
//------------------------------------------------------------------------------
void IOHIDTransactionAddElement(      
                                IOHIDTransactionRef             transaction,
                                IOHIDElementRef                 element)
{
    (*transaction->transactionInterface)->addElement(
                                        transaction->transactionInterface, 
                                        element, 
                                        0);
}
                                
//------------------------------------------------------------------------------
// IOHIDTransactionRemoveElement
//------------------------------------------------------------------------------
void IOHIDTransactionRemoveElement(
                                IOHIDTransactionRef             transaction,
                                IOHIDElementRef                 element)
{
    (*transaction->transactionInterface)->removeElement(
                                        transaction->transactionInterface, 
                                        element, 
                                        0);
}
                                
//------------------------------------------------------------------------------
// IOHIDTransactionContainsElement
//------------------------------------------------------------------------------
Boolean IOHIDTransactionContainsElement(
                                IOHIDTransactionRef             transaction,
                                IOHIDElementRef                 element)
{
    Boolean hasElement = FALSE;
    
    (*transaction->transactionInterface)->containsElement(
                                            transaction->transactionInterface, 
                                            element, 
                                            &hasElement, 
                                            0);
                                            
    return hasElement;
}
                                
//------------------------------------------------------------------------------
// IOHIDTransactionScheduleWithRunLoop
//------------------------------------------------------------------------------
void IOHIDTransactionScheduleWithRunLoop(
                                IOHIDTransactionRef             transaction, 
                                CFRunLoopRef                    runLoop, 
                                CFStringRef                     runLoopMode)
{
    if ( !transaction->asyncEventSource) {
        IOReturn ret;
        
        ret = (*transaction->transactionInterface)->getAsyncEventSource(
                                            transaction->transactionInterface,
                                            &transaction->asyncEventSource);
        
        if (ret != kIOReturnSuccess || !transaction->asyncEventSource)
            return;
    }

    transaction->runLoop     = runLoop;
    transaction->runLoopMode = runLoopMode;
        
    if (CFGetTypeID(transaction->asyncEventSource) == CFRunLoopSourceGetTypeID())
        CFRunLoopAddSource( transaction->runLoop, 
                            (CFRunLoopSourceRef)transaction->asyncEventSource, 
                            transaction->runLoopMode);
    else if (CFGetTypeID(transaction->asyncEventSource) == CFRunLoopTimerGetTypeID())
        CFRunLoopAddTimer(  transaction->runLoop, 
                            (CFRunLoopTimerRef)transaction->asyncEventSource, 
                            transaction->runLoopMode);

}

//------------------------------------------------------------------------------
// IOHIDTransactionUnscheduleFromRunLoop
//------------------------------------------------------------------------------
void IOHIDTransactionUnscheduleFromRunLoop(  
                                IOHIDTransactionRef                   transaction, 
                                CFRunLoopRef                    runLoop, 
                                CFStringRef                     runLoopMode)
{
    if ( !transaction->asyncEventSource )
        return;
        
    if (CFGetTypeID(transaction->asyncEventSource) == CFRunLoopSourceGetTypeID())
        CFRunLoopRemoveSource(  runLoop, 
                                (CFRunLoopSourceRef)transaction->asyncEventSource, 
                                runLoopMode);
    else if (CFGetTypeID(transaction->asyncEventSource) == CFRunLoopTimerGetTypeID())
        CFRunLoopRemoveTimer(   runLoop, 
                                (CFRunLoopTimerRef)transaction->asyncEventSource, 
                                runLoopMode);
                                
    transaction->runLoop     = NULL;
    transaction->runLoopMode = NULL;
}
                                
//------------------------------------------------------------------------------
// IOHIDTransactionSetValue
//------------------------------------------------------------------------------
void IOHIDTransactionSetValue(
                                IOHIDTransactionRef             transaction,
                                IOHIDElementRef                 element, 
                                IOHIDValueRef                   value,
                                IOOptionBits                    options)
{
    (*transaction->transactionInterface)->setValue(
                                            transaction->transactionInterface, 
                                            element, 
                                            value, 
                                            options);
}

//------------------------------------------------------------------------------
// IOHIDTransactionGetValue
//------------------------------------------------------------------------------
IOHIDValueRef IOHIDTransactionGetValue(
                                IOHIDTransactionRef             transaction,
                                IOHIDElementRef                 element,
                                IOOptionBits                    options)
{
    IOHIDValueRef value = NULL;
    IOReturn ret;
    
    ret = (*transaction->transactionInterface)->getValue(
                                            transaction->transactionInterface, 
                                            element, 
                                            &value, 
                                            options);
                                            
    return (ret == kIOReturnSuccess) ? value : NULL;
}
       
//------------------------------------------------------------------------------
// IOHIDTransactionCommit
//------------------------------------------------------------------------------
IOReturn IOHIDTransactionCommit(
                                IOHIDTransactionRef             transaction)
{
    return (*transaction->transactionInterface)->commit(transaction->transactionInterface, 0, NULL, NULL, 0);
}
                                
//------------------------------------------------------------------------------
// IOHIDTransactionCommitWithCallback
//------------------------------------------------------------------------------
IOReturn IOHIDTransactionCommitWithCallback(
                                IOHIDTransactionRef             transaction,
                                CFTimeInterval                  timeout, 
                                IOHIDCallback                   callback, 
                                void *                          context)
{
    IOReturn                       ret  = kIOReturnNoMemory;
    IOHIDTransactionCallbackInfo * info = (IOHIDTransactionCallbackInfo *)malloc(sizeof(IOHIDTransactionCallbackInfo));

    if (info) {
        info->context     = context;
        info->callback    = callback;
        info->transaction = transaction;
        
        ret = (*transaction->transactionInterface)->commit(transaction->transactionInterface, timeout, __IOHIDTransactionCommitCallback, info, 0);
    }

    if (ret && info) {
        free(info);
    }
    return ret;
}

//------------------------------------------------------------------------------
// IOHIDTransactionUnscheduleFromRunLoop
//------------------------------------------------------------------------------
void IOHIDTransactionClear(
                                IOHIDTransactionRef             transaction)
{
    (*transaction->transactionInterface)->clear(
                                            transaction->transactionInterface,
                                            0);
}
