/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 7, 2024.
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
#if !__has_feature(objc_arc)
#error This file requires ARC. Add the "-fobjc-arc" compiler flag for this file.
#endif

#import "config.h"
#import "_WKWebExtensionRegisteredScriptsSQLiteStore.h"

#import "CocoaHelpers.h"
#import "Logging.h"
#import "_WKWebExtensionSQLiteDatabase.h"
#import "_WKWebExtensionSQLiteHelpers.h"
#import "_WKWebExtensionSQLiteRow.h"
#import <wtf/BlockPtr.h>
#import <wtf/WeakObjCPtr.h>

#if ENABLE(WK_WEB_EXTENSIONS)

using namespace WebKit;

static const SchemaVersion currentDatabaseSchemaVersion = 1;

static NSString * const persistAcrossSessionsKey = @"persistAcrossSessions";
static NSString * const idKey = @"id";

static NSString *rowFilterStringFromRowKeys(NSArray *keys)
{
    NSMutableArray<NSString *> *escapedAndQuotedKeys = [NSMutableArray arrayWithCapacity:keys.count];

    for (NSString *key in keys) {
        NSString *keyWithSingleQuotesEscaped = [key stringByReplacingOccurrencesOfString:@"'" withString:@"''"];
        [escapedAndQuotedKeys addObject:[NSString stringWithFormat:@"'%@'", keyWithSingleQuotesEscaped]];
    }

    return [escapedAndQuotedKeys componentsJoinedByString:@","];
}

@implementation _WKWebExtensionRegisteredScriptsSQLiteStore

- (instancetype)initWithUniqueIdentifier:(NSString *)uniqueIdentifier directory:(NSString *)directory usesInMemoryDatabase:(BOOL)useInMemoryDatabase
{
    if (!(self = [super initWithUniqueIdentifier:uniqueIdentifier directory:directory usesInMemoryDatabase:useInMemoryDatabase]))
        return nil;

    return self;
}

- (void)updateScripts:(NSArray *)scripts completionHandler:(void (^)(NSString *errorMessage))completionHandler
{
    NSArray *ids = mapObjects(scripts, ^(id key, NSDictionary *script) {
        return script[idKey];
    });

    auto weakSelf = WeakObjCPtr<_WKWebExtensionRegisteredScriptsSQLiteStore> { self };
    [self deleteScriptsWithIDs:ids completionHandler:^(NSString *errorMessage) {
        auto strongSelf = weakSelf.get();
        if (!strongSelf)
            return;

        if (errorMessage.length) {
            completionHandler(errorMessage);
            return;
        }

        [strongSelf addScripts:scripts completionHandler:^(NSString *errorMessage) {
            completionHandler(errorMessage);
        }];
    }];
}

- (void)deleteScriptsWithIDs:(NSArray *)ids completionHandler:(void (^)(NSString *errorMessage))completionHandler
{
    if (!ids.count) {
        completionHandler(nil);
        return;
    }

    auto weakSelf = WeakObjCPtr<_WKWebExtensionRegisteredScriptsSQLiteStore> { self };
    dispatch_async(_databaseQueue, ^{
        auto strongSelf = weakSelf.get();
        if (!strongSelf)
            return;

        NSString *errorMessage;
        if (![strongSelf _openDatabaseIfNecessaryReturningErrorMessage:&errorMessage createIfNecessary:NO]) {
            dispatch_async(dispatch_get_main_queue(), ^{
                completionHandler(errorMessage);
            });

            return;
        }

        ASSERT(!errorMessage.length);
        ASSERT(strongSelf->_database);

        auto result = SQLiteDatabaseExecute(strongSelf->_database, [NSString stringWithFormat:@"DELETE FROM registered_scripts WHERE key in (%@)", rowFilterStringFromRowKeys(ids)]);
        if (result != SQLITE_DONE) {
            RELEASE_LOG_ERROR(Extensions, "Failed to delete scripts for extension %{private}@.", strongSelf->_uniqueIdentifier);
            errorMessage = @"Failed to delete scripts from registered content scripts storage.";
        }

        NSString *deleteDatabaseErrorMessage = [strongSelf _deleteDatabaseIfEmpty];

        dispatch_async(dispatch_get_main_queue(), ^{
            // Errors from opening the database or deleting keys take precedence over an error deleting the database.
            completionHandler(errorMessage.length ? errorMessage : deleteDatabaseErrorMessage);
        });
    });
}

- (void)addScripts:(NSArray *)scripts completionHandler:(void (^)(NSString *errorMessage))completionHandler
{
    // Only save persistent scripts to storage.
    scripts = filterObjects(scripts, ^(id key, NSDictionary *script) {
        return boolForKey(script, persistAcrossSessionsKey , false);
    });

    if (!scripts.count) {
        completionHandler(nil);
        return;
    }

    auto weakSelf = WeakObjCPtr<_WKWebExtensionRegisteredScriptsSQLiteStore> { self };
    dispatch_async(_databaseQueue, ^{
        auto strongSelf = weakSelf.get();
        if (!strongSelf)
            return;

        NSString *errorMessage;

        if (![strongSelf _openDatabaseIfNecessaryReturningErrorMessage:&errorMessage]) {
            dispatch_async(dispatch_get_main_queue(), ^{
                completionHandler(errorMessage);
            });

            return;
        }

        ASSERT(!errorMessage.length);
        ASSERT(strongSelf->_database);

        for (NSDictionary *script in scripts)
            [strongSelf _insertScript:script inDatabase:strongSelf->_database errorMessage:&errorMessage];

        dispatch_async(dispatch_get_main_queue(), ^{
            completionHandler(errorMessage);
        });
    });
}

- (void)getScriptsWithCompletionHandler:(void (^)(NSArray *, NSString *))completionHandler
{
    auto weakSelf = WeakObjCPtr<_WKWebExtensionRegisteredScriptsSQLiteStore> { self };
    dispatch_async(_databaseQueue, ^{
        NSString *errorMessage;
        auto strongSelf = weakSelf.get();
        auto* scripts = [strongSelf _getScriptsWithOutErrorMessage:&errorMessage];

        dispatch_async(dispatch_get_main_queue(), ^{
            completionHandler(scripts, errorMessage);
        });
    });
}

- (NSArray *)_getScriptsWithOutErrorMessage:(NSString **)outErrorMessage
{
    dispatch_assert_queue(_databaseQueue);

    if (![self _openDatabaseIfNecessaryReturningErrorMessage:outErrorMessage createIfNecessary:NO])
        return @[ ];

    ASSERT(!(*outErrorMessage).length);
    ASSERT(_database);

    _WKWebExtensionSQLiteRowEnumerator *rows = SQLiteDatabaseFetch(_database, @"SELECT * FROM registered_scripts");
    return [self _getKeysAndValuesFromRowEnumerator:rows];
}

- (NSArray *)_getKeysAndValuesFromRowEnumerator:(_WKWebExtensionSQLiteRowEnumerator *)rows
{
    static NSSet *allowedClasses = [NSSet setWithObjects:NSString.class, NSArray.class, NSMutableDictionary.class, nil];

    NSMutableArray<NSDictionary<NSString *, id> *> *scripts = [NSMutableArray array];

    for (_WKWebExtensionSQLiteRow *row in rows) {
        NSError *error;
        NSDictionary<NSString *, id> *script = [NSKeyedUnarchiver unarchivedObjectOfClasses:allowedClasses fromData:[row dataAtIndex:1] error:&error];

        if (error)
            RELEASE_LOG_ERROR(Extensions, "Failed to deserialize registered content script for extension %{private}@.", _uniqueIdentifier);
        else
            [scripts addObject:script];
    }

    return scripts;
}

- (void)_insertScript:(NSDictionary *)script inDatabase:(_WKWebExtensionSQLiteDatabase *)database errorMessage:(NSString **)errorMessage
{
    dispatch_assert_queue(_databaseQueue);
    ASSERT(_database);

    NSData *scriptAsData = [NSKeyedArchiver archivedDataWithRootObject:script requiringSecureCoding:YES error:nil];
    NSString *scriptID = script[idKey];
    ASSERT(scriptID);

    auto result = SQLiteDatabaseExecute(database, @"INSERT INTO registered_scripts (key, script) VALUES (?, ?)", scriptID, scriptAsData);
    if (result != SQLITE_DONE) {
        RELEASE_LOG_ERROR(Extensions, "Failed to insert registered content script for extension %{private}@.", _uniqueIdentifier);
        *errorMessage =  @"Failed to add content script.";
        return;
    }
}

// MARK: Database Schema

- (SchemaVersion)_currentDatabaseSchemaVersion
{
    return currentDatabaseSchemaVersion;
}

- (NSURL *)_databaseURL
{
    if (_useInMemoryDatabase)
        return [_WKWebExtensionSQLiteDatabase inMemoryDatabaseURL];

    ASSERT(_directory);

    NSString *databaseName = @"RegisteredContentScripts.db";

    return [_directory URLByAppendingPathComponent:databaseName isDirectory:NO];
}

- (DatabaseResult)_createFreshDatabaseSchema
{
    dispatch_assert_queue(_databaseQueue);
    ASSERT(_database);

    auto result = SQLiteDatabaseExecute(_database, @"CREATE TABLE registered_scripts (key TEXT PRIMARY KEY NOT NULL, script BLOB NOT NULL)");
    if (result != SQLITE_DONE)
        RELEASE_LOG_ERROR(Extensions, "Failed to create registered_scripts database for extension %{private}@: %{public}@ (%d)", _uniqueIdentifier, _database.lastErrorMessage, result);

    return result;
}

- (DatabaseResult)_resetDatabaseSchema
{
    dispatch_assert_queue(_databaseQueue);
    ASSERT(_database);

    auto result = SQLiteDatabaseExecute(_database, @"DROP TABLE IF EXISTS registered_scripts");
    if (result != SQLITE_DONE)
        RELEASE_LOG_ERROR(Extensions, "Failed to reset registered_scripts database schema for extension %{private}@: %{public}@ (%d)", _uniqueIdentifier, _database.lastErrorMessage, result);

    return result;
}

- (BOOL)_isDatabaseEmpty
{
    dispatch_assert_queue(_databaseQueue);
    ASSERT(_database);

    _WKWebExtensionSQLiteRowEnumerator *rows = SQLiteDatabaseFetch(_database, @"SELECT COUNT(*) FROM registered_scripts");
    return ![[rows nextObject] int64AtIndex:0];
}

@end

#endif // ENABLE(WK_WEB_EXTENSIONS)

