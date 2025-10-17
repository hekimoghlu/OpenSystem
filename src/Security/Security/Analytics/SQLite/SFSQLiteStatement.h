/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 16, 2024.
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
#ifndef SECURITY_SFSQLSTATEMENT_H
#define SECURITY_SFSQLSTATEMENT_H 1

#if __OBJC2__

#import <Foundation/Foundation.h>
#import <sqlite3.h>

@class SFSQLite;

@protocol SFSQLiteRow <NSObject>

- (NSUInteger)columnCount;
- (int)columnTypeAtIndex:(NSUInteger)index;
- (NSString *)columnNameAtIndex:(NSUInteger)index;
- (NSUInteger)indexForColumnName:(NSString *)columnName;

- (SInt32)intAtIndex:(NSUInteger)index;
- (SInt64)int64AtIndex:(NSUInteger)index;
- (double)doubleAtIndex:(NSUInteger)index;
- (NSData *)blobAtIndex:(NSUInteger)index;
- (NSString *)textAtIndex:(NSUInteger)index;
- (id)objectAtIndex:(NSUInteger)index;
- (NSArray *)allObjects;
- (NSDictionary *)allObjectsByColumnName;

@end

@interface SFSQLiteStatement : NSObject <SFSQLiteRow> {
    __weak SFSQLite* _SQLite;
    NSString* _SQL;
    sqlite3_stmt* _handle;
    BOOL _reset;
    NSMutableArray* _temporaryBoundObjects;
}

- (id)initWithSQLite:(SFSQLite *)SQLite SQL:(NSString *)SQL handle:(sqlite3_stmt *)handle;

@property (nonatomic, readonly, weak)     SFSQLite     *SQLite;
@property (nonatomic, readonly, strong)   NSString       *SQL;
@property (nonatomic, readonly, assign)   sqlite3_stmt   *handle;

@property (nonatomic, assign, getter=isReset) BOOL reset;

- (BOOL)step;
- (void)reset;

- (void)finalizeStatement;

- (void)bindInt:(SInt32)value atIndex:(NSUInteger)index;
- (void)bindInt64:(SInt64)value atIndex:(NSUInteger)index;
- (void)bindDouble:(double)value atIndex:(NSUInteger)index;
- (void)bindBlob:(NSData *)value atIndex:(NSUInteger)index;
- (void)bindText:(NSString *)value atIndex:(NSUInteger)index;
- (void)bindNullAtIndex:(NSUInteger)index;
- (void)bindValue:(id)value atIndex:(NSUInteger)index;
- (void)bindValues:(NSArray *)values;

@end

#endif /* __OBJC2__ */
#endif /* SECURITY_SFSQLSTATEMENT_H */
