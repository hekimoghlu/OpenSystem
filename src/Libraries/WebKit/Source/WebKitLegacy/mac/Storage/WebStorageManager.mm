/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 9, 2023.
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
#import "WebStorageManagerInternal.h"

#import "StorageTracker.h"
#import "WebKitVersionChecks.h"
#import "WebSecurityOriginInternal.h"
#import "WebStorageNamespaceProvider.h"
#import "WebStorageTrackerClient.h"
#import <WebCore/SecurityOrigin.h>
#import <WebCore/SecurityOriginData.h>
#import <wtf/cocoa/VectorCocoa.h>

using namespace WebCore;

NSString * const WebStorageDirectoryDefaultsKey = @"WebKitLocalStorageDatabasePathPreferenceKey";
NSString * const WebStorageDidModifyOriginNotification = @"WebStorageDidModifyOriginNotification";

@implementation WebStorageManager

+ (WebStorageManager *)sharedWebStorageManager
{
    static WebStorageManager *sharedManager = [[WebStorageManager alloc] init];
    return sharedManager;
}

- (id)init
{
    if (!(self = [super init]))
        return nil;
    
    WebKitInitializeStorageIfNecessary();
    
    return self;
}

- (NSArray *)origins
{
    return createNSArray(WebKit::StorageTracker::tracker().origins(), [] (auto&& origin) {
        return adoptNS([[WebSecurityOrigin alloc] _initWithWebCoreSecurityOrigin:origin.securityOrigin().ptr()]);
    }).autorelease();
}

- (void)deleteAllOrigins
{
    WebKit::StorageTracker::tracker().deleteAllOrigins();
#if PLATFORM(IOS_FAMILY)
    // FIXME: This needs to be removed once StorageTrackers in multiple processes
    // are in sync: <rdar://problem/9567500> Remove Website Data pane is not kept in sync with Safari
    [[NSFileManager defaultManager] removeItemAtPath:[WebStorageManager _storageDirectoryPath] error:NULL];
#endif
}

- (void)deleteOrigin:(WebSecurityOrigin *)origin
{
    WebKit::StorageTracker::tracker().deleteOrigin([origin _core]->data());
}

- (unsigned long long)diskUsageForOrigin:(WebSecurityOrigin *)origin
{
    return WebKit::StorageTracker::tracker().diskUsageForOrigin([origin _core]);
}

- (void)syncLocalStorage
{
    WebKit::WebStorageNamespaceProvider::syncLocalStorage();
}

- (void)syncFileSystemAndTrackerDatabase
{
    WebKit::StorageTracker::tracker().syncFileSystemAndTrackerDatabase();
}

+ (NSString *)_storageDirectoryPath
{
    static NeverDestroyed<RetainPtr<NSString>> sLocalStoragePath;
    static dispatch_once_t flag;
    dispatch_once(&flag, ^{
        NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
        RetainPtr<NSString> localStoragePath = [defaults objectForKey:WebStorageDirectoryDefaultsKey];
        if (!localStoragePath || ![localStoragePath isKindOfClass:[NSString class]]) {
            NSArray *paths = NSSearchPathForDirectoriesInDomains(NSLibraryDirectory, NSUserDomainMask, YES);
            NSString *libraryDirectory = [paths objectAtIndex:0];
            localStoragePath = [libraryDirectory stringByAppendingPathComponent:@"WebKit/LocalStorage"];
        }
        sLocalStoragePath.get() = [localStoragePath stringByStandardizingPath];
    });
    return sLocalStoragePath.get().get();
}

+ (void)setStorageDatabaseIdleInterval:(double)interval
{
    WebKit::StorageTracker::tracker().setStorageDatabaseIdleInterval(1_s * interval);
}

+ (void)closeIdleLocalStorageDatabases
{
    WebKit::WebStorageNamespaceProvider::closeIdleLocalStorageDatabases();
}

void WebKitInitializeStorageIfNecessary()
{
    static BOOL initialized = NO;
    if (initialized)
        return;

    auto *storagePath = [WebStorageManager _storageDirectoryPath];
    WebKit::StorageTracker::initializeTracker(storagePath, WebStorageTrackerClient::sharedWebStorageTrackerClient());

#if PLATFORM(IOS_FAMILY)
    [[NSURL fileURLWithPath:storagePath] setResourceValue:@YES forKey:NSURLIsExcludedFromBackupKey error:nil];
#endif

    initialized = YES;
}

@end
