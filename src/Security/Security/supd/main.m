/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 6, 2023.
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
#include <TargetConditionals.h>
#import <Foundation/NSError_Private.h>
#import <dirhelper_priv.h>

#if TARGET_OS_OSX
#include <sandbox.h>
#include <notify.h>
#include <pwd.h>
#endif

#if TARGET_OS_SIMULATOR

int main(int argc, char** argv)
{
    return 0;
}

#else

#import <Foundation/Foundation.h>
#import "supd.h"
#include "utilities/debugging.h"
#import <Foundation/NSXPCConnection_Private.h>
#include <xpc/private.h>

@interface ServiceDelegate : NSObject <NSXPCListenerDelegate>
@end

@implementation ServiceDelegate

- (BOOL)listener:(NSXPCListener *)listener shouldAcceptNewConnection:(NSXPCConnection *)newConnection {
    /* Client must either have the supd entitlement or the trustd file helping entitlement.
     * Each method of the protocol will additionally check for the entitlement it needs. */
    NSNumber *supdEntitlement = [newConnection valueForEntitlement:@"com.apple.private.securityuploadd"];
    BOOL hasSupdEntitlement = [supdEntitlement isKindOfClass:[NSNumber class]] && [supdEntitlement boolValue];
    NSNumber *trustdHelperEntitlement = [newConnection valueForEntitlement:@"com.apple.private.trustd.FileHelp"];
    BOOL hasTrustdHelperEntitlement = [trustdHelperEntitlement isKindOfClass:[NSNumber class]] && [trustdHelperEntitlement boolValue];

    /* expose the protocol based the client's entitlement (a client can't do both) */
    if (hasSupdEntitlement) {
        secinfo("xpc", "Client (pid: %d) properly entitled for supd interface, let's go", [newConnection processIdentifier]);
        newConnection.exportedInterface = [NSXPCInterface interfaceWithProtocol:@protocol(supdProtocol)];
    } else if (hasTrustdHelperEntitlement) {
        secinfo("xpc", "Client (pid: %d) properly entitled for trustd file helper interface, let's go", [newConnection processIdentifier]);
        newConnection.exportedInterface = [NSXPCInterface interfaceWithProtocol:@protocol(TrustdFileHelper_protocol)];
    } else {
        secerror("xpc: Client (pid: %d) doesn't have entitlement", [newConnection processIdentifier]);
        return NO;
    }

    supd *exportedObject = [[supd alloc] initWithConnection:newConnection];
    newConnection.exportedObject = exportedObject;
    [newConnection resume];
    return YES;
}

@end

static void securityuploadd_sandbox(void)
{
#if TARGET_OS_OSX
    // Enter the sandbox on macOS
    char homeDir[PATH_MAX] = {};
    char buf[PATH_MAX] = "";

    if (!_set_user_dir_suffix("com.apple.securityuploadd") ||
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

    struct passwd* pwd = getpwuid(getuid());
    if (pwd == NULL) {
        secerror("Failed to get home directory for user: %d", errno);
        exit(EXIT_FAILURE);
    }

    if (realpath(pwd->pw_dir, homeDir) == NULL) {
        strlcpy(homeDir, pwd->pw_dir, sizeof(homeDir));
    }

    const char *sandbox_params[] = {
        "HOME", homeDir,
        "_TMPDIR", tempdir,
        "_DARWIN_CACHE_DIR", cachedir,
        NULL
    };

    char *sberror = NULL;
    secerror("initializing securityuploadd sandbox with HOME=%s", homeDir);
    if (sandbox_init_with_parameters("com.apple.securityuploadd", SANDBOX_NAMED, sandbox_params, &sberror) != 0) {
        secerror("Failed to enter securityuploadd sandbox: %{public}s", sberror);
        exit(EXIT_FAILURE);
    }

    free(tempdir);
    free(cachedir);
#else // !TARGET_OS_OSX
    char buf[PATH_MAX] = "";
    _set_user_dir_suffix("com.apple.securityuploadd");
    confstr(_CS_DARWIN_USER_TEMP_DIR, buf, sizeof(buf));
#endif // !TARGET_OS_OSX
}

int main(int argc, const char *argv[])
{
    secnotice("lifecycle", "supd lives!");
    [NSError _setFileNameLocalizationEnabled:NO];
    securityuploadd_sandbox();

    ServiceDelegate *delegate = [[ServiceDelegate alloc] init];

    // Always create a supd instance to register for the background activity that doesn't check entitlements
    static supd *activity_supd = nil;
    activity_supd = [[supd alloc] initWithConnection:nil];
    
    NSXPCListener *listener = [[NSXPCListener alloc] initWithMachServiceName:@"com.apple.securityuploadd"];
    listener.delegate = delegate;

    // We're always launched in response to client activity and don't want to sit around idle.
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 5ull * NSEC_PER_SEC), dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        secnotice("lifecycle", "will exit when clean");
        xpc_transaction_exit_clean();
    });

    [listener resume];
    [[NSRunLoop currentRunLoop] run];
    return 0;
}

#endif  // !TARGET_OS_SIMULATOR
