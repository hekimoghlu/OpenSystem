/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 17, 2024.
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
#import "GSSCredXPCHelperClient.h"
#import <xpc/private.h>
#import <Foundation/NSXPCConnection_Private.h>
#import <CoreFoundation/CFXPCBridge.h>
#import "gsscred.h"
#import "common.h"

@interface GSSCredXPCHelperClient ()

@end

@implementation GSSCredXPCHelperClient

+ (NSXPCConnection *)createXPCConnection:(uid_t)session
{
    NSXPCConnection *xpcConnection;
    xpcConnection = [[NSXPCConnection alloc] initWithMachServiceName:@"com.apple.GSSCred" options:NSXPCConnectionPrivileged];
    [xpcConnection setInterruptionHandler:^{
	os_log_debug(GSSOSLog(), "connection interrupted: %u", session);
    }];
    
    [xpcConnection setInvalidationHandler:^{
	os_log_debug(GSSOSLog(), "connection invalidated: %u", session);
    }];
    
    uuid_t uuid;
    uuid_parse("D58511E6-6A96-41F0-B5CB-885DF4E3A531", uuid);  //make this value static to avoid duplicate launches of GSSCred
    if (session != 0) {
	memcpy(&uuid, &session, sizeof(session));
	xpc_connection_set_oneshot_instance(xpcConnection._xpcConnection, uuid);
    }
    
    [xpcConnection resume];
    return xpcConnection;

}

+ (void)sendWakeup:(NSXPCConnection *)connection
{
    xpc_object_t request = xpc_dictionary_create(NULL, NULL, 0);
    xpc_dictionary_set_string(request, "command", "wakeup");
    xpc_connection_send_message(connection._xpcConnection, request);
}

+ (krb5_error_code)acquireForCred:(HeimCredRef)cred expireTime:(time_t *)expire
{
    os_log_debug(GSSOSLog(), "gsscred_cache_acquire: %@", CFBridgingRelease(CFUUIDCreateString(NULL, cred->uuid)));
    
    NSXPCConnection *xpcConnection = [self createXPCConnection:cred->session];
    [self sendWakeup:xpcConnection];
    
    CFDictionaryRef attributes = cred->attributes;
    xpc_object_t xpcattrs = _CFXPCCreateXPCObjectFromCFObject(attributes);
    if (xpcattrs == NULL)
	return KRB5_FCC_INTERNAL;

    xpc_object_t request = xpc_dictionary_create(NULL, NULL, 0);
    xpc_dictionary_set_string(request, "command", "acquire");
    xpc_dictionary_set_value(request, "attributes", xpcattrs);
     
    xpc_object_t reply = xpc_connection_send_message_with_reply_sync(xpcConnection._xpcConnection, request);
    [xpcConnection invalidate];
    if (reply == NULL) {
	os_log_error(GSSOSLog(), "server did not return any data");
    }

    if (xpc_get_type(reply) == XPC_TYPE_ERROR) {
	os_log_error(GSSOSLog(), "server returned an error: %@", reply);
    }

    if (xpc_get_type(reply) == XPC_TYPE_DICTIONARY) {
	NSDictionary *replyDictionary = CFBridgingRelease(_CFXPCCreateCFObjectFromXPCObject(reply));
	
	NSDictionary *resultDictionary = replyDictionary[@"result"];
	NSNumber *status = resultDictionary[@"status"];
	NSNumber *expireTime = resultDictionary[@"expire"];
	*expire = [expireTime longValue];

	return [status intValue];
	
    }

    return 1;
}

+ (krb5_error_code)refreshForCred:(HeimCredRef)cred expireTime:(time_t *)expire
{
    os_log_debug(GSSOSLog(), "gsscred_cache_refresh: %@", CFBridgingRelease(CFUUIDCreateString(NULL, cred->uuid)));
    
    NSXPCConnection *xpcConnection = [self createXPCConnection:cred->session];
    [self sendWakeup:xpcConnection];
    
    CFDictionaryRef attributes = cred->attributes;
    xpc_object_t xpcattrs = _CFXPCCreateXPCObjectFromCFObject(attributes);
    if (xpcattrs == NULL)
	return KRB5_FCC_INTERNAL;
    
    xpc_object_t request = xpc_dictionary_create(NULL, NULL, 0);
    xpc_dictionary_set_string(request, "command", "refresh");
    xpc_dictionary_set_value(request, "attributes", xpcattrs);
    
    xpc_object_t reply = xpc_connection_send_message_with_reply_sync(xpcConnection._xpcConnection, request);
    [xpcConnection invalidate];
    if (reply == NULL) {
	os_log_error(GSSOSLog(), "server returned an error during wakeup: %@", reply);
    }

    if (xpc_get_type(reply) == XPC_TYPE_ERROR) {
	os_log_error(GSSOSLog(), "server returned an error: %@", reply);
    }

    if (xpc_get_type(reply) == XPC_TYPE_DICTIONARY) {
	NSDictionary *replyDictionary = CFBridgingRelease(_CFXPCCreateCFObjectFromXPCObject(reply));
	
	NSDictionary *resultDictionary = replyDictionary[@"result"];
	NSNumber *status = resultDictionary[@"status"];
	NSNumber *expireTime = resultDictionary[@"expire"];
	*expire = [expireTime longValue];
	
	return [status intValue];
	
    }
    
    return 1;
    
}

@end
