/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 30, 2022.
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
#import "WebDatabaseManagerPrivate.h"

#import "WebDatabaseManagerClient.h"
#import "WebDatabaseProvider.h"
#import "WebPlatformStrategies.h"
#import "WebSecurityOriginInternal.h"
#import <WebCore/DatabaseManager.h>
#import <WebCore/DatabaseTracker.h>
#import <WebCore/SecurityOrigin.h>
#import <wtf/cocoa/VectorCocoa.h>

#if PLATFORM(IOS_FAMILY)
#import "WebDatabaseManagerInternal.h"
#import <WebCore/DatabaseTracker.h>
#import <WebCore/WebCoreThread.h>
#endif

using namespace WebCore;

NSString *WebDatabaseDirectoryDefaultsKey = @"WebDatabaseDirectory";

NSString *WebDatabaseDisplayNameKey = @"WebDatabaseDisplayNameKey";
NSString *WebDatabaseExpectedSizeKey = @"WebDatabaseExpectedSizeKey";
NSString *WebDatabaseUsageKey = @"WebDatabaseUsageKey";

NSString *WebDatabaseDidModifyOriginNotification = @"WebDatabaseDidModifyOriginNotification";
NSString *WebDatabaseDidModifyDatabaseNotification = @"WebDatabaseDidModifyDatabaseNotification";
NSString *WebDatabaseIdentifierKey = @"WebDatabaseIdentifierKey";

#if PLATFORM(IOS_FAMILY)
CFStringRef WebDatabaseOriginsDidChangeNotification = CFSTR("WebDatabaseOriginsDidChangeNotification");
#endif

static NSString *databasesDirectoryPath();

@implementation WebDatabaseManager

+ (WebDatabaseManager *) sharedWebDatabaseManager
{
    static WebDatabaseManager *sharedManager = [[WebDatabaseManager alloc] init];
    return sharedManager;
}

- (id)init
{
    if (!(self = [super init]))
        return nil;

    WebPlatformStrategies::initializeIfNecessary();

    DatabaseManager& dbManager = DatabaseManager::singleton();

    // Set the database root path in WebCore
    dbManager.initialize(databasesDirectoryPath());

    // Set the DatabaseManagerClient
    dbManager.setClient(&WebKit::WebDatabaseManagerClient::sharedWebDatabaseManagerClient());

    return self;
}

- (NSArray *)origins
{
    return createNSArray(DatabaseTracker::singleton().origins(), [] (auto&& origin) {
        return adoptNS([[WebSecurityOrigin alloc] _initWithWebCoreSecurityOrigin:origin.securityOrigin().ptr()]);
    }).autorelease();
}

- (NSArray *)databasesWithOrigin:(WebSecurityOrigin *)origin
{
    if (!origin)
        return nil;
    return createNSArray(DatabaseTracker::singleton().databaseNames([origin _core]->data())).autorelease();
}

- (NSDictionary *)detailsForDatabase:(NSString *)databaseIdentifier withOrigin:(WebSecurityOrigin *)origin
{
    if (!origin)
        return nil;

    auto details = DatabaseManager::singleton().detailsForNameAndOrigin(databaseIdentifier, *[origin _core]);
    if (details.name().isNull())
        return nil;

    return @{
        WebDatabaseDisplayNameKey: details.displayName().isEmpty() ? databaseIdentifier : (NSString *)details.displayName(),
        WebDatabaseExpectedSizeKey: @(details.expectedUsage()),
        WebDatabaseUsageKey: @(details.currentUsage()),
    };
}

- (void)deleteAllDatabases
{
    DatabaseTracker::singleton().deleteAllDatabasesImmediately();
#if PLATFORM(IOS_FAMILY)
    // FIXME: This needs to be removed once DatabaseTrackers in multiple processes
    // are in sync: <rdar://problem/9567500> Remove Website Data pane is not kept in sync with Safari
    [[NSFileManager defaultManager] removeItemAtPath:databasesDirectoryPath() error:NULL];
#endif
}

- (BOOL)deleteOrigin:(WebSecurityOrigin *)origin
{
    return origin && DatabaseTracker::singleton().deleteOrigin([origin _core]->data());
}

- (BOOL)deleteDatabase:(NSString *)databaseIdentifier withOrigin:(WebSecurityOrigin *)origin
{
    return origin && DatabaseTracker::singleton().deleteDatabase([origin _core]->data(), databaseIdentifier);
}

// For DumpRenderTree support only
- (void)deleteAllIndexedDatabases
{
    WebDatabaseProvider::singleton().deleteAllDatabases();
}

#if PLATFORM(IOS_FAMILY)

static bool isFileHidden(NSString *file)
{
    ASSERT([file length]);
    return [file characterAtIndex:0] == '.';
}

+ (void)removeEmptyDatabaseFiles
{
    NSString *databasesDirectory = databasesDirectoryPath();
    NSFileManager *fileManager = [NSFileManager defaultManager];
    NSArray *array = [fileManager contentsOfDirectoryAtPath:databasesDirectory error:0];
    if (!array)
        return;
    
    NSUInteger count = [array count];
    for (NSUInteger i = 0; i < count; ++i) {
        NSString *fileName = [array objectAtIndex:i];
        // Skip hidden files.
        if (![fileName length] || isFileHidden(fileName))
            continue;
        
        NSString *path = [databasesDirectory stringByAppendingPathComponent:fileName];
        // Look for directories that contain database files belonging to the same origins.
        BOOL isDirectory;
        if (![fileManager fileExistsAtPath:path isDirectory:&isDirectory] || !isDirectory)
            continue;
        
        // Make sure the directory is not a symbolic link that points to something else.
        NSDictionary *attributes = [fileManager attributesOfItemAtPath:path error:0];
        if ([attributes fileType] == NSFileTypeSymbolicLink)
            continue;
        
        NSArray *databaseFilesInOrigin = [fileManager contentsOfDirectoryAtPath:path error:0];
        NSUInteger databaseFileCount = [databaseFilesInOrigin count];
        NSUInteger deletedDatabaseFileCount = 0;
        for (NSUInteger j = 0; j < databaseFileCount; ++j) {
            NSString *dbFileName = [databaseFilesInOrigin objectAtIndex:j];
            // Skip hidden files.
            if (![dbFileName length] || isFileHidden(dbFileName))
                continue;
            
            NSString *dbFilePath = [path stringByAppendingPathComponent:dbFileName];
            
            // There shouldn't be any directories in this folder - but check for it anyway.
            if (![fileManager fileExistsAtPath:dbFilePath isDirectory:&isDirectory] || isDirectory)
                continue;
            
            if (DatabaseTracker::deleteDatabaseFileIfEmpty(dbFilePath))
                ++deletedDatabaseFileCount;
        }
        
        // If we have removed every database file for this origin, delete the folder for this origin.
        if (databaseFileCount == deletedDatabaseFileCount || ![fileManager contentsOfDirectoryAtPath:path error:nullptr].count) {
            // Use rmdir - we don't want the deletion to happen if the folder is not empty.
            rmdir([path fileSystemRepresentation]);
        }
    }
}

+ (void)scheduleEmptyDatabaseRemoval
{
    DatabaseTracker::emptyDatabaseFilesRemovalTaskWillBeScheduled();
    
    dispatch_async(dispatch_get_global_queue(0, 0), ^{
        [WebDatabaseManager removeEmptyDatabaseFiles];
        DatabaseTracker::emptyDatabaseFilesRemovalTaskDidFinish();
    });
}

#endif // PLATFORM(IOS_FAMILY)

@end

static NSString *databasesDirectoryPath()
{
    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
    NSString *databasesDirectory = [defaults objectForKey:WebDatabaseDirectoryDefaultsKey];
    if (!databasesDirectory || ![databasesDirectory isKindOfClass:[NSString class]])
        databasesDirectory = @"~/Library/WebKit/Databases";
    
    return [databasesDirectory stringByStandardizingPath];
}
