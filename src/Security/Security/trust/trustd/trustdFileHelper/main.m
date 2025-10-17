/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 3, 2021.
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
//  main.m
//  trustdFileHelper
//
//  Copyright Â© 2020 Apple Inc. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <Foundation/NSError_Private.h>
#import <Foundation/NSXPCConnection_Private.h>
#import <Security/SecTask.h>

#import <xpc/private.h>
#import <xpc/xpc.h>
#import <xpc/activity_private.h>

#import <dirhelper_priv.h>
#include <sandbox.h>
#include <utilities/SecCFWrappers.h>
#include <utilities/SecFileLocations.h>

#include "trust/trustd/trustdFileLocations.h"
#include "trust/trustd/trustdFileHelper/trustdFileHelper.h"

@interface ServiceDelegate : NSObject <NSXPCListenerDelegate>
@end

@implementation ServiceDelegate

- (BOOL)listener:(NSXPCListener *)listener shouldAcceptNewConnection:(NSXPCConnection *)newConnection
{
    if(![[newConnection valueForEntitlement:@"com.apple.private.trustd.FileHelp"] boolValue]) {
        SecTaskRef clientTask = SecTaskCreateWithAuditToken(NULL, [newConnection auditToken]);
        secerror("rejecting client %@ due to lack of entitlement", clientTask);
        CFReleaseNull(clientTask);
        return NO;
    }

    secdebug("ipc", "opening connection for %d", [newConnection processIdentifier]);
    newConnection.exportedInterface = [NSXPCInterface interfaceWithProtocol:@protocol(TrustdFileHelper_protocol)];
    TrustdFileHelper *exportedObject = [[TrustdFileHelper alloc] init];
    newConnection.exportedObject = exportedObject;
    [newConnection resume];
    return YES;
}

@end

static void enter_sandbox(void) {
    char buf[PATH_MAX] = "";

    if (!_set_user_dir_suffix("com.apple.trustd") ||
        confstr(_CS_DARWIN_USER_TEMP_DIR, buf, sizeof(buf)) == 0 ||
        (mkdir(buf, 0700) && errno != EEXIST)) {
        secerror("failed to initialize temporary directory (%d): %s", errno, strerror(errno));
        exit(EXIT_FAILURE);
    }

    char *tempdir = realpath(buf, NULL);
    if (tempdir == NULL) {
        secerror("failed to resolve temporary directory (%d): %s", errno, strerror(errno));
        exit(EXIT_FAILURE);
    }

    if (confstr(_CS_DARWIN_USER_CACHE_DIR, buf, sizeof(buf)) == 0 ||
        (mkdir(buf, 0700) && errno != EEXIST)) {
        secerror("failed to initialize cache directory (%d): %s", errno, strerror(errno));
        exit(EXIT_FAILURE);
    }

    char *cachedir = realpath(buf, NULL);
    if (cachedir == NULL) {
        secerror("failed to resolve cache directory (%d): %s", errno, strerror(errno));
        exit(EXIT_FAILURE);
    }

    const char *parameters[] = {
        "_TMPDIR", tempdir,
        "_DARWIN_CACHE_DIR", cachedir,
        NULL
    };

    char *sberror = NULL;
    if (sandbox_init_with_parameters("com.apple.trustdFileHelper", SANDBOX_NAMED, parameters, &sberror) != 0) {
        secerror("Failed to enter trustdFileHelper sandbox: %{public}s", sberror);
        exit(EXIT_FAILURE);
    }

    free(tempdir);
    free(cachedir);
}

int
main(int argc, const char *argv[])
{
    [NSError _setFileNameLocalizationEnabled:NO];
    enter_sandbox();

    /* daemon mode */
    static NSXPCListener *listener = nil;

    ServiceDelegate *delegate = [[ServiceDelegate alloc] init];
    listener = [[NSXPCListener alloc] initWithMachServiceName:@TrustdFileHelperXPCServiceName];
    listener.delegate = delegate;

    [listener resume];
    secdebug("ipc", "trustdFileHelper accepting work");

    dispatch_main();
}
