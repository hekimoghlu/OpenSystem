/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 15, 2024.
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
#include "IOHIDEventSystem.h"
#include "IOHIDWorkLoop.h"
#include "IOHIDDebug.h"
#include <libkern/c++/OSBoundedArrayRef.h>


typedef struct _EventServiceInfo 
{
    IOHIDEventService * service;

} EventServiceInfo, * EventServiceInfoRef;

typedef struct _HIDEventArgs
{
    void *              refCon;
    AbsoluteTime        timeStamp;
    UInt32              eventCount;
    IOHIDEvent *        events;
    IOOptionBits        options;
} HIDEventArgs, * HIDEventArgsRef;


#define super IOService
OSDefineMetaClassAndStructors(IOHIDEventSystem, IOService)


//====================================================================================================
// IOHIDEventService::init
//====================================================================================================
bool IOHIDEventSystem::init(OSDictionary * properties)
{
    if ( super::init(properties) == false )
        return false;
        
    _eventServiceInfoArray = OSArray::withCapacity(4);
        
    return true;
    
}

//====================================================================================================
// IOHIDEventService::start
//====================================================================================================
bool IOHIDEventSystem::start(IOService * provider)
{
    if ( super::start(provider) == false )
        return false;
        
    _workLoop       = IOHIDWorkLoop::workLoop();
    _commandGate    = IOCommandGate::commandGate(this);
    
    if ( !_workLoop || !_commandGate )
        return false;
        
    if ( _workLoop->addEventSource(_commandGate) != kIOReturnSuccess )
        return false;

    _publishNotify = addNotification( 
                        gIOPublishNotification, 
                        serviceMatching("IOHIDEventService"),
                        OSMemberFunctionCast(IOServiceNotificationHandler, this, &IOHIDEventSystem::notificationHandler),
                        this, 
                        (void *)OSMemberFunctionCast(IOCommandGate::Action, this, &IOHIDEventSystem::handleServicePublicationGated) );

    _terminateNotify = addNotification( 
                        gIOTerminatedNotification, 
                        serviceMatching("IOHIDEventService"),
                        OSMemberFunctionCast(IOServiceNotificationHandler, this, &IOHIDEventSystem::notificationHandler),
                        this, 
                        (void *)OSMemberFunctionCast(IOCommandGate::Action, this, &IOHIDEventSystem::handleServiceTerminationGated) );

    if (!_publishNotify || !_terminateNotify) 
        return false;
        
    _eventsOpen = true;
    
    registerService();
    
    return true;
}

//====================================================================================================
// IOHIDEventService::free
//====================================================================================================
void IOHIDEventSystem::free()
{
    if (workLoop) {
        workLoop->disableAllEventSources();
    }
    
    if ( _publishNotify ) {
        _publishNotify->remove();
        _publishNotify = 0;
    }
    
    if ( _terminateNotify ) {
        _terminateNotify->remove();
        _terminateNotify = 0;
    }
    
    if ( _eventServiceInfoArray ) {
        _eventServiceInfoArray->release();
        _eventServiceInfoArray = 0;
    }
    
    if ( _commandGate ) {
        _commandGate->release();
        _commandGate = 0;
    }
    
    if ( _workLoop ) {
        _workLoop->release();
        _workLoop = 0;
    }
    super::free();
}

//====================================================================================================
// IOHIDEventService::message
//====================================================================================================
IOReturn IOHIDEventSystem::message(UInt32 type, IOService * provider, void * argument)
{
    return super::message(type, provider, argument);
}

//====================================================================================================
// IOHIDEventService::setProperties
//====================================================================================================
IOReturn IOHIDEventSystem::setProperties( OSObject * properties )
{
    return super::setProperties(properties);
}

//====================================================================================================
// IOHIDEventService::notificationHandler
//====================================================================================================
bool IOHIDEventSystem::notificationHandler( void * refCon,  IOService * service )
{
    HIDLog("");

    _commandGate->runAction((IOCommandGate::Action)refCon, service);
    
    return true;
}

//====================================================================================================
// IOHIDEventService::handleServicePublicationGated
//====================================================================================================
void IOHIDEventSystem::handleServicePublicationGated(IOService * service)
{
    HIDLog("");

    EventServiceInfo    tempEventServiceInfo;
    OSData *            tempData;
    IOHIDEventService * eventService;
    
    if ( !(eventService = OSDynamicCast(IOHIDEventService, service)) )
        return;
    
    attach( eventService );

    tempEventServiceInfo.service = eventService;
    
    tempData = OSData::withBytes(&tempEventServiceInfo, sizeof(EventServiceInfo));
    
    if ( tempData )
    {
        _eventServiceInfoArray->setObject(tempData);
        tempData->release();
    }
    
    if ( _eventsOpen )
        registerEventSource( eventService );
        
}

//====================================================================================================
// IOHIDEventService::handleServiceTerminationGated
//====================================================================================================
void IOHIDEventSystem::handleServiceTerminationGated(IOService * service)
{
    EventServiceInfoRef tempEventServiceInfoRef;
    OSData *            tempData;
    UInt32              index;

    HIDLog("");
    
    if ( _eventsOpen )
        service->close(this);

    for ( index = 0; index<_eventServiceInfoArray->getCount(); index++ )
    {
        if ( (tempData = OSDynamicCast(OSData, _eventServiceInfoArray->getObject(index)))
            && (tempEventServiceInfoRef = (EventServiceInfoRef)tempData->getBytesNoCopy())
            && (tempEventServiceInfoRef->service == service) )
        {
            _eventServiceInfoArray->removeObject(index);
            break;
        }
    }
        
    detach(service);
}

//====================================================================================================
// IOHIDEventService::registerEventSource
//====================================================================================================
void IOHIDEventSystem::registerEventSource(IOHIDEventService * service)
{
    EventServiceInfoRef tempEventServiceInfoRef;
    OSData *            tempData = NULL;
    UInt32              index;

    HIDLog("");

    for ( index = 0; index<_eventServiceInfoArray->getCount(); index++ )
    {
        if ( (tempData = OSDynamicCast(OSData, _eventServiceInfoArray->getObject(index)))
            && (tempEventServiceInfoRef = (EventServiceInfoRef)tempData->getBytesNoCopy())
            && (tempEventServiceInfoRef->service == service) )
            break;
        
        tempData = NULL;
    }

    service->open(this, 0, tempData, 
                OSMemberFunctionCast(IOHIDEventService::HIDEventCallback, this, &IOHIDEventSystem::handleHIDEvent));
}


//====================================================================================================
// IOHIDEventService::handleHIDEvent
//====================================================================================================
void IOHIDEventSystem::handleHIDEvent(
                            void *                          refCon,
                            AbsoluteTime                    timeStamp,
                            UInt32                          eventCount,
                            IOHIDEvent *                    events,
                            IOOptionBits                    options)
{
    HIDLog("");
    
    HIDEventArgs args;
    
    args.refCon     = refCon;
    args.timeStamp  = timeStamp;
    args.eventCount = eventCount;
    args.events     = events;
    args.options    = options;
    
    _commandGate->runAction(OSMemberFunctionCast(IOCommandGate::Action, this, &IOHIDEventSystem::handleHIDEventGated), (void *)&args);
}

//====================================================================================================
// IOHIDEventService::handleHIDEventGated
//====================================================================================================
void IOHIDEventSystem::handleHIDEventGated(void * args)
{    
    HIDEventArgsRef eventArgsRef = (HIDEventArgsRef)args;
    
    if ( !eventArgsRef->events ) 
    {
        HIDLogError("type=%d timestamp=%lld", 0, *((UInt64 *)&(eventArgsRef->timeStamp)));
        return;
    }
    
    HIDLog("eventCount=%d timestamp=%lld", eventArgsRef->eventCount, *((UInt64 *)&(eventArgsRef->timeStamp)));

    OSBoundedArrayRef<IOHIDEvent> events(&(eventArgsRef->events[0]), eventArgsRef->eventCount);
    for ( IOHIDEvent event : events )
    {
        HIDLog("type=%d", event.type);
        switch ( event.type )
        {
            case kIOHIDKeyboardEvent:
                HIDLog(" usagePage=%x usage=%x value=%d repeat=%d", event.data.keyboard.usagePage, event.data.keyboard.usage, event.data.keyboard.value, event.data.keyboard.repeat);
                break;
                
            case kIOHIDMouseEvent:
                HIDLog(" buttons=%x dx=%d dy=%d", event.data.mouse.buttons, event.data.mouse.dx, event.data.mouse.dy);
                break;
                
            case kIOHIDScrollEvent:
                HIDLog(" deltaAxis1=%d deltaAxis2=%d deltaAxis3=%d", event.data.scroll.lines.deltaAxis1, event.data.scroll.lines.deltaAxis2, event.data.scroll.lines.deltaAxis3);
                break;
                
            default:
                break;
        }
    }
    
}
