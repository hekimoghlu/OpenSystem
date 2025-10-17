/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 3, 2023.
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
//  _PMCoreSmartPowerNap.m
//  LowPowerMode-Embedded
//
//  Created by Prateek Malhotra on 12/7/22.
//

#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>
#import <libproc.h>
#import <os/log.h>
#import <os/transaction_private.h>
#import <notify.h>
#import <IOKit/pwr_mgt/IOPMLibPrivate.h>
#import "_PMCoreSmartPowerNap.h"

static os_log_t coresmartpowernap_log = NULL;
#define LOG_STREAM  coresmartpowernap_log

#define ERROR_LOG(fmt, args...) \
{  \
    if (coresmartpowernap_log) \
    os_log_error(LOG_STREAM, fmt, ##args); \
    else\
    os_log_error(OS_LOG_DEFAULT, fmt, ##args); \
}

#define INFO_LOG(fmt, args...) \
{  \
    if (coresmartpowernap_log) \
    os_log(LOG_STREAM, fmt, ##args); \
    else \
    os_log(OS_LOG_DEFAULT, fmt, ##args); \
}

#define FAULT_LOG(fmt, args...) \
{  \
    os_log_fault(LOG_STREAM, fmt, ##args); \
}

NSString *const kPMCoreSmartPowerNapServiceName = @"com.apple.powerd.coresmartpowernap";

@implementation _PMCoreSmartPowerNap

- (instancetype)init {
    self = [super init];
    //setup logging
    coresmartpowernap_log = os_log_create("com.apple.powerd", "coresmartpowernap");

    // establish connection
    if (self) {
        _connection = [[NSXPCConnection alloc]initWithMachServiceName:kPMCoreSmartPowerNapServiceName options:NSXPCConnectionPrivileged];
        _connection.remoteObjectInterface = [NSXPCInterface interfaceWithProtocol:@protocol(_PMCoreSmartPowerNapProtocol)];
        _connection.exportedObject = self;
        _connection.exportedInterface = [NSXPCInterface interfaceWithProtocol:@protocol(_PMCoreSmartPowerNapCallbackProtocol)];

        // interruption handler
        __weak typeof(self) welf = self;
        [_connection setInterruptionHandler:^{
            typeof(self) client = welf;
            if (!client) {
                return;
            }
            INFO_LOG("Connection to powerd interrupted");
            client.connection_interrupted = YES;
        }];

        [_connection setInvalidationHandler:^{
            typeof(self) client = welf;
            if (!client) {
                return;
            }
            INFO_LOG("Connection to powerd invalidated");
        }];

        [_connection resume];
        _connection_interrupted = NO;
        INFO_LOG("Initialized connection");

        // Register to re-establish connection on powerd's restart
        static int resync_token;
        int status = notify_register_dispatch(kIOUserAssertionReSync, &resync_token, dispatch_get_main_queue(), ^(int token __unused) {
            typeof(self) client = welf;
            if (!client) {
                return;
            }
            if (client.connection_interrupted) {
                INFO_LOG("Powerd has restarted");
                [client reRegister];
                client.connection_interrupted = NO;
            }
        });
        if (status != NOTIFY_STATUS_OK) {
            ERROR_LOG("Failed to register for reconnect with powerd 0x%x", status);
        }
    }
    return self;
}

- (void)registerWithCallback:(dispatch_queue_t)queue callback:(_PMCoreSmartPowerNapCallback)callback {
    pid_t my_pid = getpid();
    char name[128];
    proc_name(my_pid, name, sizeof(name));
    _identifier = [NSString stringWithFormat:@"%@:%s", [[NSUUID UUID] UUIDString], name];
    [self registerWithIdentifier:_identifier];
    _callback_queue = queue;
    _callback = callback;
}

- (_PMCoreSmartPowerNapState)state {
    return _current_state;
}

- (void)registerWithIdentifier:(NSString *)identifier {
    if (identifier) {
        INFO_LOG("registerWithIdentifier %@", identifier);
        [[_connection remoteObjectProxyWithErrorHandler:^(NSError * _Nonnull error) {
                ERROR_LOG("Failed to register %@", error);
        }] registerWithIdentifier:identifier];
    } else {
        FAULT_LOG("Failed to register. Expected non-null identifier");
    }
}

- (void)reRegister {
    INFO_LOG("re-register with powerd with identifier: %@", _identifier);
    [self registerWithIdentifier:_identifier];
}

- (void)unregister {
    if (_identifier) {
        [self unregisterWithIdentifier:_identifier];
    } else {
        ERROR_LOG("unregister called without registering");
        FAULT_LOG("unregister called without registering. No identifier found");
    }
}

- (void)unregisterWithIdentifier:(NSString *)identifier {
    INFO_LOG("unregisterWithIdentifier %@", identifier);
    [[_connection remoteObjectProxyWithErrorHandler:^(NSError * _Nonnull error) {
                ERROR_LOG("Failed to unregister %@", error);
    }]unregisterWithIdentifier:identifier];
}

- (void)updateState:(_PMCoreSmartPowerNapState)state {
    // callback on queue supplied
    _current_state = state;
    NS_VALID_UNTIL_END_OF_SCOPE os_transaction_t transaction = os_transaction_create("com.apple.powerd.coresmartpowernap.callback");
    if (_callback && _callback_queue) {
        dispatch_async(_callback_queue, ^{
            (void)transaction;
            self->_callback(state);
        });
    }
}

- (void)setState:(_PMCoreSmartPowerNapState)state {
    INFO_LOG("Updating CSPN state to %d", (int)state);
    [[_connection remoteObjectProxyWithErrorHandler:^(NSError * _Nonnull error) {
                ERROR_LOG("Failed to update CSPN state %@", error);
    }]setState:state];
}

- (void)setCSPNQueryDelta:(uint32_t)seconds {
    INFO_LOG("Updating CSPN Query delay");
    [[_connection remoteObjectProxyWithErrorHandler:^(NSError * _Nonnull error) {
            ERROR_LOG("Failed to CSPN Re-query delay %@", error);
    }]setCSPNQueryDelta:seconds];
}

- (void)setCSPNRequeryDelta:(uint32_t)seconds {
    INFO_LOG("Updating CSPN Re-query delay");
    [[_connection remoteObjectProxyWithErrorHandler:^(NSError * _Nonnull error) {
            ERROR_LOG("Failed to CSPN Re-query delay %@", error);
    }]setCSPNRequeryDelta:seconds];
}

- (void)setCSPNIgnoreRemoteClient:(uint32_t)state {
    INFO_LOG("Updating setCSPNIgnoreRemoteClient state to %d", (int)state);
    [[_connection remoteObjectProxyWithErrorHandler:^(NSError * _Nonnull error) {
                ERROR_LOG("Failed to update setCSPNIgnoreRemoteClient %@", error);
    }]setCSPNIgnoreRemoteClient:state];
}

- (void)setCSPNMotionAlarmThreshold:(uint32_t)seconds {
    INFO_LOG("Updating CSPN motion alarm threshold");
    [[_connection remoteObjectProxyWithErrorHandler:^(NSError * _Nonnull error) {
            ERROR_LOG("Failed to CSPN Re-entry delay %@", error);
    }]setCSPNMotionAlarmThreshold:seconds];
}

- (void)setCSPNMotionAlarmStartThreshold:(uint32_t)seconds {
    INFO_LOG("Updating CSPN motion alarm start threshold");
    [[_connection remoteObjectProxyWithErrorHandler:^(NSError * _Nonnull error) {
            ERROR_LOG("Failed to CSPN Re-entry delay %@", error);
    }]setCSPNMotionAlarmStartThreshold:seconds];
}

- (void) syncStateWithHandler:(_PMCoreSmartPowerNapCallback)handler {
    [[_connection remoteObjectProxyWithErrorHandler:^(NSError * _Nonnull error) {
        ERROR_LOG("syncStateWithHandler failed %@", error);
    }] syncStateWithHandler:handler];
}

- (_PMCoreSmartPowerNapState)syncState {
    __block _PMCoreSmartPowerNapState new_state = _PMCoreSmartPowerNapStateOff;
    [[_connection synchronousRemoteObjectProxyWithErrorHandler:^(NSError * _Nonnull error) {
        ERROR_LOG("syncState synchronous update failed %@", error);
    }]syncStateWithHandler:^(_PMCoreSmartPowerNapState state) {
        new_state = state;
        self.current_state = new_state;
    }];
    return new_state;
}
@end


