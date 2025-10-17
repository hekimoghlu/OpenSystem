/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 10, 2023.
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
#import <CoreFoundation/CoreFoundation.h>
#import "DADisk.h"
#import "DAFileSystem.h"
#import "DAMain.h"
#import <Foundation/Foundation.h>
#import "DALog.h"
#if TARGET_OS_OSX || TARGET_OS_IOS
#import <FSKit/FSKit.h>
#import <FSKit/FSKitDiskArbHelper_private.h>
#import <FSKit/FSAuditToken_private.h>
#import <UserFS/LiveFSUSBLocalStorageClient.h>
#import <LiveFS/LiveFSMountManagerClient.h>
#import <LiveFS/LiveFS_LiveFileMounter.h>
#import <os/log.h>
#import <os/feature_private.h>


int __DAMountUserFSVolume( void * parameter )
{
    int returnValue                            = 0;
    __DAFileSystemContext *context             = parameter;
    FSAuditToken          *token;
    
    NSString *fsType       = (__bridge NSString *)context->fileSystem;
    NSString *deviceName   = (__bridge NSString *)context->deviceName;
    NSString *mountpoint   = (__bridge NSString *)context->mountPoint;
    NSString *volumeName   = (__bridge NSString *)context->volumeName;
    NSString *mountOptions = (__bridge NSString *)context->mountOptions;

    if ( [FSClient class] == nil )
    {
        // No FSKit in the run time, bail
        DALogError("Attempt to use FSKit, when not present, to mount volume of type %@",
                   fsType);
        returnValue = EPROTONOSUPPORT;
        goto exit;
    }

    token = [FSAuditToken new];
    token = [token tokenWithRuid:gDAConsoleUserUID];
    
    returnValue = [FSKitDiskArbHelper DAMountUserFSVolume:fsType
                                               deviceName:deviceName
                                               mountPoint:mountpoint
                                               volumeName:volumeName
                                               auditToken:token.audit_token
                                             mountOptions:mountOptions];

exit:
    
    return returnValue;
}

int DAUserFSOpen( char *path, int flags )
{
    int fd = -1;


    xpc_connection_t server = xpc_connection_create_mach_service( "com.apple.filesystems.userfs_helper", NULL, 0 );
    assert( server != NULL );
    assert( xpc_get_type(server) == XPC_TYPE_CONNECTION );
    xpc_connection_set_event_handler( server, ^(xpc_object_t object) { /* do nothing */ } );
    xpc_connection_resume( server );

    xpc_object_t msg = xpc_dictionary_create( NULL, NULL, 0 );
    xpc_dictionary_set_string( msg, "path", path );
    xpc_dictionary_set_int64 (msg, "flags", flags );

    xpc_object_t reply = xpc_connection_send_message_with_reply_sync( server, msg );
    if ( reply != NULL && xpc_get_type(reply) == XPC_TYPE_DICTIONARY )
    {
        fd = xpc_dictionary_dup_fd( reply, "fd" );
        if ( fd < 0 )
        {
            int64_t error = xpc_dictionary_get_int64( reply, "error" );
            DALogInfo( "open:error:%d", (int)error );
            if ( error == 0 )
            {
                error = EIO;
            }
            errno = (int)error;
        }
    }
    else
    {
        DALogInfo( "open:invalidReply:%{public}s", reply ? xpc_copy_description(reply) : "NULL" );
        errno = EIO;
    }
    
    xpc_connection_cancel( server );

    return fd;
}
#endif
