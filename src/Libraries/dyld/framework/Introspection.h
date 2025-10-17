/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 27, 2024.
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
#ifndef DyldIntrospection_h
#define DyldIntrospection_h

#import <Foundation/Foundation.h>

// DO NOT USE ANYTHING N THIS FILE
// THESE SPIs WILL BE CHANGING

NS_HEADER_AUDIT_BEGIN(nullability, sendability)

@interface _DYSegment : NSObject
@property(nonatomic, readonly) NSString* name;
@property(nonatomic, readonly) uint64_t vmsize;
@property(nonatomic, readonly) uint64_t permissions;
@property(nonatomic, readonly) uint64_t address;
@property(nonatomic, readonly) uint64_t preferredLoadAddress;
- (BOOL)withSegmentData:(void (^_Nonnull)(NSData * _Nonnull))block;
@end


@class _DYSharedCache;
@class _DYSnapshot;

struct _DYImageFastPathData {
    void* installNamePtr;
    uint64_t    installNameSize;
    void*       filePathPtr;
    uint64_t    filePathSize;
    uuid_t      uuid;
    uint64_t    address;
    bool        sharedCacheImage;
    bool        unicodeInstallname;
    bool        unicodeFilePath;
};

@interface _DYEnvironment : NSObject
@property(nonatomic, readonly, nullable) NSString* rootPath;
@end

@class _DYSharedCache;
@class _DYSnapshot;

@interface _DYImage : NSObject
@property(nonatomic, readonly, nullable) NSString* installname;
@property(nonatomic, readonly, nullable) NSString* filePath;
@property(nonatomic, readonly, nullable) NSUUID* uuid;
@property(nonatomic, readonly) uint64_t address;
@property(nonatomic, readonly) uint64_t preferredLoadAddress;
@property(nonatomic, readonly) NSArray<_DYSegment *>* segments;
@property(nonatomic, readonly, nullable) _DYSharedCache* sharedCache;
@property(nonatomic, readonly) uint64_t pointerSize; //FIXME: This should be on the process, but needs to be here to support

- (void) getFastPathData:(struct _DYImageFastPathData*)data;
@end

//FIXME: These should be properties of _DYImage, but the way they are implemented in `dyld` makes that inconvenient for now
@interface _DYAOTImage : NSObject
@property(nonatomic, readonly) uint64_t x86Address;
@property(nonatomic, readonly) uint64_t aotAddress;
@property(nonatomic, readonly) uint64_t aotSize;
@property(nonatomic, readonly) NSData* aotImageKey;
@end

// FIXME: Only here to support dyld_shared_cache_for_each_subcache4Rosetta
@interface _DYSubCache : NSObject
- (BOOL)withVMLayoutData:(void (^_Nonnull)(NSData * _Nonnull))block;
@end

@interface _DYSharedCache : NSObject
@property(nonatomic, readonly) NSUUID* uuid;
@property(nonatomic, readonly) uint64_t aotAddress;
@property(nonatomic, readonly, nullable) NSUUID* aotUuid;
@property(nonatomic, readonly) uint64_t address;
@property(nonatomic, readonly) uint64_t vmsize;
@property(nonatomic, readonly) uint64_t preferredLoadAddress;
@property(nonatomic, readonly) BOOL mappedPrivate;
@property(nonatomic, readonly) NSArray<_DYImage *>* images;
@property(nonatomic, readonly) NSArray<NSString *>* filePaths;
@property(nonatomic, readonly) NSArray<_DYSubCache *>* subCaches; // FIXME: Only here to support dyld_shared_cache_for_each_subcache4Rosetta
@property(nonatomic, readonly, nullable) NSString* localSymbolPath;
@property(nonatomic, readonly, nullable) NSData* localSymbolData;
- (BOOL)pinMappings;
- (void)unpinMappings;

+ (NSArray<_DYSharedCache *> *) installedSharedCaches;
+ (NSArray<_DYSharedCache *> *) installedSharedCachesForSystemPath:(NSString*)path NS_SWIFT_NAME(installedSharedCaches(systemPath:));
- (instancetype _Nullable) initWithPath:(NSString*)path error:(NSError  * _Nullable * _Nullable)error;
@end

@interface _DYSnapshot : NSObject

- (instancetype _Nullable)initWithData:(NSData * _Nonnull)data error:(NSError  * _Nullable * _Nullable)error;

@property(nonatomic, readonly) uint64_t platform; // FIXME: Should be dyld_platform_t
@property(nonatomic, readonly) size_t pageSize;
@property(nonatomic, readonly) uint64_t timestamp;
@property(nonatomic, readonly) uint64_t initialImageCount;
@property(nonatomic, readonly) uint8_t state;
@property(nonatomic, readonly) pid_t pid;
@property(nonatomic, readonly) NSArray<_DYImage *>* images;
@property(nonatomic, readonly, nullable) NSArray<_DYAOTImage *>* aotImages;
@property(nonatomic, readonly, nullable) _DYSharedCache* sharedCache;
@property(nonatomic, readonly, nullable) _DYEnvironment* environment;
@end

@interface _DYEventHandlerToken : NSObject
@property(nonatomic, readonly) uint32_t value;
- (instancetype) initWithValue:(uint32_t) value;
@end

@interface _DYProcess : NSObject
+ (instancetype) processForCurrentTask;
- (instancetype _Nullable) initWithTask:(task_read_t)task queue:(_Nullable dispatch_queue_t)queue error:(NSError  * _Nullable * _Nullable)error NS_SWIFT_NAME(init(task:queue:));
- (_DYSnapshot* _Nullable) getCurrentSnapshotAndReturnError:(NSError  * _Nullable * _Nullable)error NS_SWIFT_NAME(getCurrentSnapshot());
- (_DYEventHandlerToken* _Nullable)registerChangeNotificationsWithError:(NSError  * _Nullable * _Nullable)error handler:(void (^)(_DYImage* image, bool load))block NS_SWIFT_NAME(registerChangeNotifications(handler:));
- (_DYEventHandlerToken* _Nullable)registerForEvent:(uint32_t)event error:(NSError  * _Nullable * _Nullable)error handler:(void (^)())block NS_SWIFT_NAME(register(event:handler:));
@property(nonatomic, nullable) dispatch_queue_t queue;

- (void)unregisterForEvent:(_DYEventHandlerToken*)event NS_SWIFT_NAME(unregister(event:));
@end

//
NS_HEADER_AUDIT_END(nullability, sendability)

#endif /* DyldIntrospection_h */
