/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 23, 2024.
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
// Header exposed for unit testing only

#ifndef SECURITY_SFSQL_H
#define SECURITY_SFSQL_H 1

#if __OBJC2__

#import <Foundation/Foundation.h>
#import <sqlite3.h>

@class SFSQLiteStatement;

typedef SInt64 SFSQLiteRowID;
@class SFSQLite;
@protocol SFSQLiteRow;

NSArray *SFSQLiteJournalSuffixes(void);

typedef NS_ENUM(NSInteger, SFSQLiteSynchronousMode) {
    SFSQLiteSynchronousModeOff = 0,
    SFSQLiteSynchronousModeNormal = 1, // default
    SFSQLiteSynchronousModeFull = 2
};

@protocol SFSQLiteDelegate <NSObject>
@property (nonatomic, readonly) SInt32 userVersion;

- (BOOL)migrateDatabase:(SFSQLite *)db fromVersion:(SInt32)version;
@end

// Wrapper around the SQLite API. Typically subclassed to add table accessor methods.
@interface SFSQLite : NSObject {
@private
    id<SFSQLiteDelegate> _delegate;
    NSString* _path;
    NSString* _schema;
    NSString* _schemaVersion;
    NSMutableDictionary* _statementsBySQL;
    NSString* _objectClassPrefix;
    SFSQLiteSynchronousMode _synchronousMode;
    SInt32 _userVersion;
    sqlite3* _db;
    NSUInteger _openCount;
    NSDateFormatter* _dateFormatter;
#if DEBUG
    NSMutableDictionary* _unitTestOverrides;
#endif
    BOOL _hasMigrated;
    BOOL _corrupt;
    BOOL _traced;
}

- (instancetype)initWithPath:(NSString *)path schema:(NSString *)schema;
    
@property (nonatomic, readonly, strong) NSString   *path;
@property (nonatomic, readonly, strong) NSString   *schema;
@property (nonatomic, readonly, strong) NSString   *schemaVersion;
@property (nonatomic, strong)           NSString   *objectClassPrefix;
@property (nonatomic, assign)           SInt32     userVersion;
@property (nonatomic, assign)           SFSQLiteSynchronousMode synchronousMode;
@property (nonatomic, readonly)         BOOL       isOpen;
@property (nonatomic, readonly)         BOOL       hasMigrated;
@property (nonatomic, assign)           BOOL       traced;

@property (nonatomic, strong) id<SFSQLiteDelegate> delegate;

#if DEBUG
@property (nonatomic, strong) NSDictionary* unitTestOverrides;
#endif

// Open/close the underlying database file read/write. Initially, the database is closed.
- (void)open;
- (BOOL)openWithError:(NSError **)error;
- (void)close;

// Remove the database file.
- (void)remove;

// Database exclusive transaction operations.
- (void)begin;
- (void)end;
- (void)rollback;

// Database maintenance.
- (void)analyze;
- (void)vacuum;

// The rowID assigned to the last record inserted into the database.
- (SFSQLiteRowID)lastInsertRowID;

// returns the number of rows modified, inserted or deleted by the most recently completed INSERT, UPDATE or DELETE statement on the database connection
- (int)changes;

// Execute one-or-more queries. Use prepared statements for anything performance critical.
- (BOOL)executeSQL:(NSString *)SQL;

// Prepared statement pool accessors. Statements must be reset after they're used.
- (SFSQLiteStatement *)statementForSQL:(NSString *)SQL;
- (void)removeAllStatements;

// Accessors for all the tables created in the database.
- (NSArray *)allTableNames;
- (void)dropAllTables;

// Generic key/value properties set in the database.
- (NSString *)propertyForKey:(NSString *)key;
- (void)setProperty:(NSString *)value forKey:(NSString *)key;
- (NSDate *)datePropertyForKey:(NSString *)key;
- (void)setDateProperty:(NSDate *)value forKey:(NSString *)key;
- (void)removePropertyForKey:(NSString *)key;

// Date the cache was created.
- (NSDate *)creationDate;

// Convience calls that generate and execute statements.
- (NSArray *)selectAllFrom:(NSString *)tableName where:(NSString *)whereSQL bindings:(NSArray *)bindings;
- (NSArray<NSDictionary *> *)select:(NSArray *)columns from:(NSString *)tableName;
- (NSArray *)select:(NSArray *)columns from:(NSString *)tableName mapEachRow:(id (^)(id<SFSQLiteRow> row))block;
- (NSArray<NSDictionary*> *)select:(NSArray*)columns from:(NSString *)tableName where:(NSString *)whereSQL bindings:(NSArray *)bindings;
- (void)select:(NSArray *)columns from:(NSString *)tableName where:(NSString *)whereSQL bindings:(NSArray *)bindings orderBy:(NSArray *)orderBy limit:(NSNumber *)limit block:(void (^)(NSDictionary *resultDictionary, BOOL *stop))block;
- (void)select:(NSArray *)columns from:(NSString *)tableName where:(NSString *)whereSQL bindings:(NSArray *)bindings orderBy:(NSArray *)orderBy limit:(NSNumber *)limit forEachRow:(void (^)(id<SFSQLiteRow> row, BOOL *stop))block;
- (void)selectFrom:(NSString *)tableName where:(NSString *)whereSQL bindings:(NSArray *)bindings orderBy:(NSArray *)orderBy limit:(NSNumber *)limit block:(void (^)(NSDictionary *resultDictionary, BOOL *stop))block;
- (NSUInteger)selectCountFrom:(NSString *)tableName  where:(NSString *)whereSQL bindings:(NSArray *)bindings;
- (SFSQLiteRowID)insertOrReplaceInto:(NSString *)tableName values:(NSDictionary *)valuesByColumnName;
- (void)deleteFrom:(NSString *)tableName where:(NSString *)whereSQL bindings:(NSArray *)bindings;
- (void)update:(NSString *)tableName set:(NSString *)setSQL where:(NSString *)whereSQL bindings:(NSArray *)whereBindings limit:(NSNumber *)limit;
- (void)deleteFrom:(NSString *)tableName matchingValues:(NSDictionary *)valuesByColumnName;
- (NSSet<NSString*> *)columnNamesForTable:(NSString*)tableName;

- (SInt32)dbUserVersion;

@end

#endif /* __OBJC2__ */
#endif /* SECURITY_SFSQL_H */
