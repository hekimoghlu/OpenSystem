/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 28, 2025.
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
#import <Foundation/Foundation.h>
#import "IOHIDDebug.h"
#import <HID/HID_Private.h>
#import <IOKit/hid/IOHIDUsageTables.h>
#import <IOKit/hid/IOHIDServiceKeys.h>
#import <IOKit/hid/IOHIDEventTypes.h>
#import <IOKit/hid/IOHIDLibPrivate.h>
#include <xpc/private.h>

@interface IOHIDTestSessionFilter : NSObject <HIDSessionFilter>

- (nullable instancetype)initWithSession:(HIDSession *)session;

- (nullable id)propertyForKey:(NSString *)key;

- (BOOL)setProperty:(nullable id)value
             forKey:(NSString *)key;

- (nullable HIDEvent *)filterEvent:(HIDEvent *)event
                        forService:(HIDEventService *)service;

- (void)activate;

- (void)setDispatchQueue:(dispatch_queue_t)queue;

- (nullable HIDEvent *)filterEvent:(HIDEvent *)event
                      toConnection:(HIDConnection *)connection
                       fromService:(HIDEventService *)service;

@end

@implementation IOHIDTestSessionFilter {
    NSNumber *_prop;
    HIDBlock _cancelHandler;
    dispatch_queue_t _queue;
    bool _activated;
    bool _clientAdded;
    bool _audit;
    NSMutableArray * _testServices;
    NSMutableDictionary * _testServicesNotifications;
    NSMutableArray * _testServicesNotificationHistory;
}

- (nullable instancetype)initWithSession:(HIDSession *)session
{
    self = [super init];
    if (!self) {
        return self;
    }
        
    _testServices = [NSMutableArray new];
    
    _testServicesNotifications = [NSMutableDictionary new];
    
    _testServicesNotificationHistory = [NSMutableArray new];
    
    HIDLog("IOHIDTestSessionFilter::initWithSession: %@", session);
    
    return self;
}

- (void)dealloc
{
    HIDLog("IOHIDTestSessionFilter dealloc");
}

- (id)propertyForKey:(NSString *)key
{
    id result = nil;
    
    if ([key isEqualToString:@"TestHIDServiceFilterGetProperty"]) {
        result = _prop;
        HIDLog("HIDServiceFilterExample::propertyForKey %@ value: %@", key, result);
    } else if ([key isEqualToString:@"TestHIDServiceFilterEnableAudit"]) {
        result = [NSNumber numberWithBool:_audit];
        HIDLog("IOHIDTestSessionFilter::propertyForKey %@ value: %@", key, result);
    } else if ([key isEqualToString:@(kIOHIDServiceFilterDebugKey)]) {
        // debug dictionary that gets captured by hidutil
        NSMutableDictionary *debug = [NSMutableDictionary new];
        
        debug[@"FilterName"] = @"IOHIDTestSessionFilter";
        debug[@"cancelHandler"] = _cancelHandler ? @YES : @NO;
        debug[@"dispatchQueue"] = _queue ? @YES : @NO;
        debug[@"activated"] = @(_activated);
        debug[@"clientAdded"] = @(_clientAdded);
        
        result = debug;
    } else if ([key isEqualToString:@("TestRequestTerminateNotificationHistory")]) {
        NSArray *testServicesNotificationHistory = nil;
        @synchronized (_testServicesNotificationHistory) {
            testServicesNotificationHistory = [_testServicesNotificationHistory copy];
        }
        result = testServicesNotificationHistory;
    } else if ([key isEqualToString:@("TestRequestTerminate")]) {
        NSMutableArray * testServices = [NSMutableArray new];
        for (id object in _testServices) {
            HIDEventService *service = (HIDEventService *)object;
            [testServices addObject: @(service.serviceID)];
        }
        result = testServices;
    }

    return result;
}

- (BOOL)setProperty:(id)value
             forKey:(NSString *)key
{
    bool result = false;
    
    if ([key isEqualToString:@"TestHIDServiceFilterSetProperty"] &&
        [value isKindOfClass:[NSNumber class]]) {
        _prop = value;
        result = true;
        
        HIDLog("IOHIDTestSessionFilter::setProperty: %@ forKey: %@", value, key);
    } else if ([key isEqualToString:@"TestHIDServiceFilterEnableAudit"]) {
        _audit = [[NSNumber numberWithBool:YES] isEqual:value];
        result = true;
        HIDLog("IOHIDTestSessionFilter::setProperty: %@ forKey: %@", value, key);
    } else if ([key isEqualToString:@"TestRequestTerminate"]) {
        HIDLog("IOHIDTestSessionFilter::setProperty: %@ forKey: %@", value, key);
        for (id object in _testServices) {
            HIDEventService *service = (HIDEventService *)object;
            HIDLog("IOHIDServiceRequestTerminate: %@", service);
            IOHIDServiceRequestTerminate((IOHIDServiceRef)service);
        }
    }
    
    return result;
}

- (nullable HIDEvent *)filterEvent:(HIDEvent *)event
               forService:(HIDEventService *)service
{
    return event;
}

- (nullable HIDEvent *)filterEvent:(HIDEvent *)event
                      toConnection:(HIDConnection *)connection
                       fromService:(HIDEventService *)service
{
//    if (connection) {
//        audit_token_t conn_token = {0};
//        xpc_object_t entitlements = nil;
//
//        [connection getAuditToken:&conn_token];
//
//        entitlements = xpc_copy_entitlement_for_token(NULL, &conn_token);
//        
//        if (entitlements) {
//            if (xpc_dictionary_get_value(entitlements, "com.apple.private.hid.testconnection.audit") != XPC_BOOL_TRUE) {
//                event = nil;
//            }
//            HIDLog("IOHIDTestSessionFilter::filterEvent entitlements: %s",  xpc_copy_description(entitlements));
//        }
//    }
//
//    HIDLog("IOHIDTestSessionFilter::filterEvent: %@ toConnection: %@ fromService: %@", event, connection, service);

    return event;
}

- (void)activate
{
    _activated = true;
}

- (void)setDispatchQueue:(dispatch_queue_t)queue
{
    _queue = queue;
}



static void IOHIDServiceReuestTeminateCallback(void * _Nullable target, void * _Nullable  refcon __unused, IOHIDServiceRef service)
{
    IOHIDTestSessionFilter * me =  (__bridge IOHIDTestSessionFilter *) target;
    [me serviceRequestTeminateNotification:(__bridge HIDEventService *)service];
}

- (void)serviceNotification:(HIDEventService *)service added:(BOOL)added
{
    id requestTerminate = [service propertyForKey:@"TestRequestTerminate"];

    if (requestTerminate) {
        HIDLog("IOHIDTestSessionFilter::serviceNotification track request terminate for %@", service);
        if (added) {
            [_testServices addObject:service];
            IOHIDNotificationRef notification =  IOHIDServiceCreateRequestTerminationNotification((__bridge IOHIDServiceRef)service, IOHIDServiceReuestTeminateCallback,  (__bridge void *)self, NULL);
            _testServicesNotifications[ @([service serviceID])] = CFBridgingRelease(notification);
        }
    }
    if (NO == added) {
        [_testServices removeObject:service];
        [_testServicesNotifications removeObjectForKey:@([service serviceID])];
    }
}

- (void)serviceRequestTeminateNotification:(HIDEventService *)service
{
    
    HIDLog("IOHIDTestSessionFilter::serviceRequestTeminateNotification %@", service);
    @synchronized (_testServicesNotificationHistory) {
        [_testServicesNotificationHistory addObject:@(service.serviceID)];
    }
}


@end
