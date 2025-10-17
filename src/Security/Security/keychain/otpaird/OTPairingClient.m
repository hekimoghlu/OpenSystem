/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 24, 2022.
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

#import <TargetConditionals.h>

#if TARGET_OS_WATCH

#import <Foundation/Foundation.h>
#import <xpc/xpc.h>
#import <os/log.h>
#import <Security/SecXPCHelper.h>

#import "keychain/categories/NSError+UsefulConstructors.h"

#import "OTPairingClient.h"
#import "OTPairingConstants.h"

void
OTPairingInitiateWithCompletion(dispatch_queue_t queue, bool immediate, void (^completion)(bool success, NSError *))
{
    xpc_connection_t connection;
    xpc_object_t message;

    connection = xpc_connection_create_mach_service(OTPairingMachServiceName, NULL, XPC_CONNECTION_MACH_SERVICE_PRIVILEGED);
    xpc_connection_set_event_handler(connection, ^(__unused xpc_object_t event) {
    });
    xpc_connection_activate(connection);

    message = xpc_dictionary_create(NULL, NULL, 0);
    xpc_dictionary_set_uint64(message, OTPairingXPCKeyOperation, OTPairingOperationInitiate);
    xpc_dictionary_set_bool(message, OTPairingXPCKeyImmediate, immediate);

    xpc_connection_send_message_with_reply(connection, message, queue, ^(xpc_object_t reply) {
        if (xpc_get_type(reply) == XPC_TYPE_DICTIONARY) {
            bool success = xpc_dictionary_get_bool(reply, OTPairingXPCKeySuccess);
            size_t errlen = 0;
            const void *errptr = xpc_dictionary_get_data(reply, OTPairingXPCKeyError, &errlen);
            NSData *errordata;
            NSError *nserr = nil;

            if (errptr != NULL) {
                errordata = [NSData dataWithBytesNoCopy:(void *)errptr length:errlen freeWhenDone:NO];
                nserr = [SecXPCHelper errorFromEncodedData:errordata];
            }

            completion(success, nserr);
        } else if (reply == XPC_ERROR_CONNECTION_INVALID) {
            // This error is expected; otpaird is a NanoLaunchDaemon, only loaded when companion is paired.
            os_log(OS_LOG_DEFAULT, "otpaird connection invalid (daemon unloaded?)");
            completion(true, nil);
        } else {
            char *desc = NULL;
            NSDictionary *userInfo = nil;

            desc = xpc_copy_description(reply);
            userInfo = @{ NSLocalizedDescriptionKey : [NSString stringWithUTF8String:desc] };
            free(desc);

            completion(false, [NSError errorWithDomain:OTPairingErrorDomain code:OTPairingErrorTypeXPC userInfo:userInfo]);
        }
    });
}

#endif /* TARGET_OS_WATCH */
