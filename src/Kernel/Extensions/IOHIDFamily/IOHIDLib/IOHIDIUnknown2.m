/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 30, 2023.
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
//  IOHIDIUnknown2.m
//  IOHIDLib
//
//  Created by dekom on 11/14/17.
//

#import <Foundation/Foundation.h>
#import <IOKit/hid/IOHIDDevicePlugIn.h>
#import "IOHIDIUnknown2.h"

@implementation IOHIDIUnknown2

static HRESULT _queryInterface(void *iunknown,
                              REFIID uuidBytes,
                              LPVOID *outInterface)
{
    IOHIDIUnknown2 *me = (__bridge id)((*((IUnknownVTbl**)iunknown))->_reserved);
    
    return [me queryInterface:uuidBytes outInterface:outInterface];
}

- (HRESULT)queryInterface:(REFIID __unused)uuidBytes
             outInterface:(LPVOID * __unused)outInterface
{
    return E_NOINTERFACE;
}

static ULONG _addRef(void *iunknown)
{
    IOHIDIUnknown2 *me = (__bridge id)((*((IUnknownVTbl**)iunknown))->_reserved);
    
    CFIndex rc = CFGetRetainCount((CFTypeRef)me);
    CFRetain((CFTypeRef)me);
    return (ULONG)(rc+1);
}

static ULONG _release(void *iunknown)
{
    IOHIDIUnknown2 *me = (__bridge id)((*((IUnknownVTbl**)iunknown))->_reserved);
    
    CFIndex rc = CFGetRetainCount((CFTypeRef)me);
    CFRelease((CFTypeRef)me);
    return (ULONG)(rc-1);
}

-(instancetype)init
{
    self = [super init];
    
    if (!self) {
        return nil;
    }
    
    _vtbl = (IUnknownVTbl *)malloc(sizeof(*_vtbl));
    
    *_vtbl = (IUnknownVTbl) {
        ._reserved = (__bridge void *)self,
        .QueryInterface = _queryInterface,
        .AddRef = _addRef,
        .Release = _release,
    };
    
    return self;
}

-(void)dealloc
{
    free(_vtbl);
}

@end

@implementation IOHIDPlugin

static IOReturn _probe(void *iunknown,
                      CFDictionaryRef properties,
                      io_service_t service,
                      SInt32 *outScore)
{
    IOHIDPlugin *me = (__bridge id)((*((IUnknownVTbl**)iunknown))->_reserved);
    
    return [me probe:(__bridge NSDictionary *)properties
             service:service
            outScore:outScore];
}

- (IOReturn)probe:(NSDictionary * _Nonnull __unused)properties
          service:(io_service_t __unused)service
         outScore:(SInt32 * _Nonnull __unused)outScore
{
    return kIOReturnUnsupported;
}

static IOReturn _start(void *iunknown,
                      CFDictionaryRef properties,
                      io_service_t service)
{
    IOHIDPlugin *me = (__bridge id)((*((IUnknownVTbl**)iunknown))->_reserved);
    
    return [me start:(__bridge NSDictionary *)properties service:service];
}

- (IOReturn)start:(NSDictionary * _Nonnull __unused)properties
          service:(io_service_t __unused)service
{
    return kIOReturnUnsupported;
}

static IOReturn _stop(void *iunknown)
{
    IOHIDPlugin *me = (__bridge id)((*((IUnknownVTbl**)iunknown))->_reserved);
    
    return [me stop];
}

- (IOReturn)stop
{
    return kIOReturnUnsupported;
}

-(instancetype)init
{
    self = [super init];
    
    if (!self) {
        return nil;
    }
    
    _plugin = (IOCFPlugInInterface *)malloc(sizeof(*_plugin));
    
    *_plugin = (IOCFPlugInInterface) {
        // IUNKNOWN_C_GUTS
        ._reserved = (__bridge void *)self,
        .QueryInterface = self->_vtbl->QueryInterface,
        .AddRef = self->_vtbl->AddRef,
        .Release = self->_vtbl->Release,
        
        // IOCFPLUGINBASE
        .version = 1,
        .revision = 0,
        .Probe = _probe,
        .Start = _start,
        .Stop = _stop,
    };
    
    CFPlugInAddInstanceForFactory(kIOHIDDeviceFactoryID);
    
    return self;
}

-(void)dealloc
{
    CFPlugInRemoveInstanceForFactory(kIOHIDDeviceFactoryID);
    free(_plugin);
}

@end
