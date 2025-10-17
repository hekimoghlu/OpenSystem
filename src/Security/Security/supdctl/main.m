/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 2, 2025.
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
#include "lib/SecArgParse.h"
#import "supd/supdProtocol.h"
#import <Foundation/NSXPCConnection_Private.h>
#import <Security/SFAnalytics.h>
#import "SecInternalReleasePriv.h"

/* Internal Topic Names */
NSString* const SFAnalyticsTopicKeySync = @"KeySyncTopic";

static void nsprintf(NSString *fmt, ...) NS_FORMAT_FUNCTION(1, 2);
static void nsprintf(NSString *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    NSString *str = [[NSString alloc] initWithFormat:fmt arguments:ap];
    va_end(ap);

    puts([str UTF8String]);
#if !__has_feature(objc_arc)
    [str release];
#endif
}

static NSXPCConnection* getConnection(void)
{
    NSXPCConnection* connection = [[NSXPCConnection alloc] initWithMachServiceName:@"com.apple.securityuploadd" options:0];
    connection.remoteObjectInterface = [NSXPCInterface interfaceWithProtocol:@protocol(supdProtocol)];
    [connection resume];
    return connection;
}

static void getSysdiagnoseDump(void)
{
    dispatch_semaphore_t sema = dispatch_semaphore_create(0);
    NSXPCConnection* connection = getConnection();
    [[connection remoteObjectProxyWithErrorHandler:^(NSError * _Nonnull error) {
        nsprintf(@"Could not communicate with supd: %@", error);
        dispatch_semaphore_signal(sema);
    }] getSysdiagnoseDumpWithReply:^(NSString * sysdiagnoseString) {
        nsprintf(@"Analytics sysdiagnose: \n%@", sysdiagnoseString);
        dispatch_semaphore_signal(sema);
    }];

    if(dispatch_semaphore_wait(sema, dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC * 20)) != 0) {
        printf("\n\nError: timed out waiting for response from supd\n");
    }
    [connection invalidate];
}

static void createLoggingJSON(char *topicName)
{
    NSString *topic = topicName ? [NSString stringWithUTF8String:topicName] : SFAnalyticsTopicKeySync;
    dispatch_semaphore_t sema = dispatch_semaphore_create(0);
    NSXPCConnection* connection = getConnection();
    [[connection remoteObjectProxyWithErrorHandler:^(NSError * _Nonnull error) {
        nsprintf(@"Could not communicate with supd: %@", error);
        dispatch_semaphore_signal(sema);
    }] createLoggingJSON:YES topic:topic reply:^(NSData* data, NSError* error) {
        if (data) {
            // Success! Only print the JSON blob to make output easier to parse
            nsprintf(@"%@", [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding]);
        } else {
            nsprintf(@"supd gave us an error: %@", error);
        }
        dispatch_semaphore_signal(sema);
    }];

    if(dispatch_semaphore_wait(sema, dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC * 20)) != 0) {
        printf("\n\nError: timed out waiting for response from supd\n");
    }
    [connection invalidate];
}

static void createChunkedLoggingJSON(char *topicName)
{
    NSString *topic = topicName ? [NSString stringWithUTF8String:topicName] : SFAnalyticsTopicKeySync;
    dispatch_semaphore_t sema = dispatch_semaphore_create(0);
    NSXPCConnection* connection = getConnection();
    [[connection remoteObjectProxyWithErrorHandler:^(NSError * _Nonnull error) {
        nsprintf(@"Could not communicate with supd: %@", error);
        dispatch_semaphore_signal(sema);
    }] createChunkedLoggingJSON:YES topic:topic reply:^(NSData* data, NSError* error) {
        if (data) {
            // Success! Only print the JSON blob to make output easier to parse
            nsprintf(@"%@", [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding]);
        } else {
            nsprintf(@"supd gave us an error: %@", error);
        }
        dispatch_semaphore_signal(sema);
    }];

    if(dispatch_semaphore_wait(sema, dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC * 20)) != 0) {
        printf("\n\nError: timed out waiting for response from supd\n");
    }
    [connection invalidate];
}

static void forceUploadAnalytics(void)
{
    dispatch_semaphore_t sema = dispatch_semaphore_create(0);
    NSXPCConnection* connection = getConnection();
    [[connection remoteObjectProxyWithErrorHandler:^(NSError * _Nonnull error) {
        nsprintf(@"Could not communicate with supd: %@", error);
        dispatch_semaphore_signal(sema);
    }] forceUploadWithReply:^(BOOL success, NSError *error) {
        if (success) {
            printf("Supd reports successful upload\n");
        } else {
            nsprintf(@"Supd reports failure: %@", error);
        }
        dispatch_semaphore_signal(sema);
    }];

    if(dispatch_semaphore_wait(sema, dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC * 20)) != 0) {
        printf("\n\nError: timed out waiting for response from supd\n");
    }
    [connection invalidate];
}

static void
getInfoDump(void)
{
    dispatch_semaphore_t sema = dispatch_semaphore_create(0);
    NSXPCConnection* connection = getConnection();
    [[connection remoteObjectProxyWithErrorHandler:^(NSError * _Nonnull error) {
        nsprintf(@"Could not communicate with supd: %@", error);
        dispatch_semaphore_signal(sema);
    }] clientStatus:^(NSDictionary<NSString *,id> *info, NSError *error) {
        if (info) {
            nsprintf(@"%@\n", info);
        } else {
            nsprintf(@"Supd reports failure: %@", error);
        }
        dispatch_semaphore_signal(sema);
    }];

    if(dispatch_semaphore_wait(sema, dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC * 20)) != 0) {
        printf("\n\nError: timed out waiting for response from supd\n");
    }
    [connection invalidate];
}

static void
forceOldUploadDate(void)
{
    dispatch_semaphore_t sema = dispatch_semaphore_create(0);
    NSXPCConnection* connection = getConnection();

    NSDate *date = [NSDate dateWithTimeIntervalSinceNow:(-7 * 24 * 3600.0)];

    [[connection remoteObjectProxyWithErrorHandler:^(NSError * _Nonnull error) {
        nsprintf(@"Could not communicate with supd: %@", error);
        dispatch_semaphore_signal(sema);
    }] setUploadDateWith:date reply:^(BOOL success, NSError *error) {
        if (!success && error) {
            nsprintf(@"Supd reports failure: %@", error);
        }
        dispatch_semaphore_signal(sema);
    }];

    if(dispatch_semaphore_wait(sema, dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC * 20)) != 0) {
        printf("\n\nError: timed out waiting for response from supd\n");
    }
    [connection invalidate];
}

static void
encodeSFACollection(NSString *jsonFile)
{
    NSData *data = [NSData dataWithContentsOfFile:jsonFile];
    NSError *error = nil;

    if (data == NULL) {
        fprintf(stderr, "file have no data: %s", [jsonFile UTF8String]);
        exit(1);
    }

    NSData *encoded = [SFAnalytics encodeSFACollection:data error:&error];
    if (encoded == NULL) {
        fprintf(stderr, "error: %s\n", [[error description] UTF8String]);
        exit(1);
    }
    fwrite(encoded.bytes, encoded.length, 1, stdout);
    return;
}


static int forceUpload = false;
static int getJSON = false;
static int getChunkedJSON = false;
static int getSysdiagnose = false;
static int getInfo = false;
static int setOldUploadDate = false;
static int sfaCollection = false;
static char *topicName = NULL;
static char *inputJsonFile = NULL;

int main(int argc, char **argv)
{
    static struct argument options[] = {
        { .shortname='t', .longname="topicName", .argument=&topicName, .description="Operate on a non-default topic"},
        { .shortname='j', .longname="jsonFile", .argument=&inputJsonFile, .description="Input JSON file"},
        { .command="sysdiagnose", .flag=&getSysdiagnose, .flagval=true, .description="Retrieve the current sysdiagnose dump for security analytics"},
        { .command="get", .flag=&getJSON, .flagval=true, .description="Get the JSON blob we would upload to the server if an upload were due"},
        { .command="getChunked", .flag=&getChunkedJSON, .flagval=true, .description="Chunk the JSON blob"},
        { .command="upload", .flag=&forceUpload, .flagval=true, .description="Force an upload of analytics data to server (ignoring privacy settings)"},
        { .command="info", .flag=&getInfo, .flagval=true, .description="Request info about clients"},
        { .command="set-old-upload-date", .flag=&setOldUploadDate, .flagval=true, .description="Clear last upload date"},
        { .command="encode-sfa-collection", .flag=&sfaCollection, .flagval=true, .description="Encode SFA Collection"},

        {}  // Need this!
    };

    static struct arguments args = {
        .programname="supdctl",
        .description="Control and report on security analytics",
        .arguments = options,
    };

    if(!options_parse(argc, argv, &args)) {
        printf("\n");
        print_usage(&args);
        return -1;
    }

    if (!SecIsInternalRelease()) {
        abort();
    }

    @autoreleasepool {
        if (forceUpload) {
            forceUploadAnalytics();
        } else if (getJSON) {
            createLoggingJSON(topicName);
        } else if (getChunkedJSON) {
            createChunkedLoggingJSON(topicName);
        } else if (getSysdiagnose) {
            getSysdiagnoseDump();
        } else if (getInfo) {
            getInfoDump();
        } else if (setOldUploadDate) {
            forceOldUploadDate();
        } else if (sfaCollection) {
            if (inputJsonFile == NULL) {
                print_usage(&args);
                return -1;
            }
            NSString *str = [NSString stringWithUTF8String:inputJsonFile];
            encodeSFACollection(str);
        } else {
            print_usage(&args);
            return -1;
        }
    }
    return 0;
}

