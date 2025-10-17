/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 2, 2022.
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
#ifndef FATMANAGER_h
#define FATMANAGER_h

#import "FATVolume.h"
#import "utils.h"

#define EOF_CLUSTER (0xFFFFFFFF)
#define EOF_RANGE_START (0xFFFFFFF8)
#define EOF_RANGE_END (0xFFFFFFFF)

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(uint8_t, iterateClustersResult) {
    iterateClustersContinue,
    iterateClustersStop
};

@interface FATBlock :NSObject

-(instancetype)initWithOffset:(uint64_t)offset;

@property NSMutableData *data;
@property uint64_t startOffset;

@end

@interface FATManager : NSObject

@property FSBlockDeviceResource *device;
@property dispatch_queue_t fatQueue;
@property FileSystemInfo *fsInfo;
@property FSOperations *fsOps;

@property uint32_t fatSize;
@property uint32_t rwSize;
@property bool useCache;

-(instancetype _Nullable)initWithDevice:(FSBlockDeviceResource *)device
                                   info:(FileSystemInfo *)info
                                    ops:(FSOperations *)fsOps
                             usingCache:(bool)usingCache;

/**
 Allocate clusters to extend an item's existing cluster chain.
 @param numOfClusters How many clusters to allocate
 @param theItem The item to allocate more clusters for
 @param allowPartial If false, failing to fully allocate clusters will return an error
 @param zeroFill CURRENTLY IGNORED
 @param mustBeContig If true, only allocate a single cluster chain.
 @param reply In case of a failure returns a non-nil error and zeros. Else, error is nil and the first, last and count of clusters allocated is returned.
 */
-(void)allocateClusters:(uint32_t)numOfClusters
                forItem:(FATItem *)theItem
           allowPartial:(bool)allowPartial
           mustBeContig:(bool)mustBeContig
               zeroFill:(bool)zeroFill
           replyHandler:(void (^)(NSError * _Nullable error,
                                  uint32_t firstAllocatedCluster,
                                  uint32_t lastAllocatedCluster,
                                  uint32_t numAllocated)) reply;

/**
 Allocate clusters wrapper starting from the FAT's first free cluster.
 @param numOfClusters How many clusters to allocate
 @param allowPartial If false, failing to fully allocate clusters will return an error
 @param zeroFill CURRENTLY IGNORED
 @param mustBeContig If true, only allocate a single cluster chain.
 @param reply In case of a failure returns a non-nil error and zeros. Else, error is nil and the first, last and count of clusters allocated is returned.
 */
-(void)allocateClusters:(uint32_t)numOfClusters
           allowPartial:(bool)allowPartial
               zeroFill:(bool)zeroFill
           mustBeContig:(bool)mustBeContig
           replyHandler:(void (^)(NSError * _Nullable error,
                                  uint32_t firstAllocatedCluster,
                                  uint32_t lastAllocatedCluster,
                                  uint32_t numAllocated)) reply;

-(void)freeClusters:(uint32_t)numClusters
             ofItem:(FATItem *)theItem
       replyHandler:(void (^)(NSError * _Nullable error)) reply;

/** THIS METHOD SHOULD ONLY BE USED IN ERROR FLOWS!
 For example, if during create, clusters were allocated but the item was not
 properly created.
 For other usages, Use freeClusters:ofFile:reply.

 Frees the cluster chain starting from startCluster up to EOF.
 Note: This method takes no item, so the FATManager won't be setting a new EOF
 to the item to which the clusters were allocated, so iterating its cluster chain
 may result in corruption. */
-(void)freeClusterFrom:(uint32_t)startCluster
           numClusters:(uint32_t)numClusters
          replyHandler:(void(^)(NSError * _Nullable error))reply;

-(void)getContigClusterChainLengthStartingAt:(uint32_t)startCluster
                                replyHandler:(void (^)(NSError * _Nullable error,
                                                       uint32_t numOfContigClusters,
                                                       uint32_t nextCluster))reply;

-(void)clusterChainLength:(FATItem*)item
             replyHandler:(void (^)(NSError * _Nullable error,
                                    uint32_t lastCluster,
                                    uint32_t length))reply;

-(void)iterateClusterChainOfItem:(FATItem *)item
                    replyHandler:(iterateClustersResult (^)(NSError * _Nullable error,
                                                            uint32_t startCluster,
                                                            uint32_t numOfContigClusters))reply;

-(void)setDirtyBitValue:(dirtyBitValue)newValue
           replyHandler:(void (^)(NSError * _Nullable error))reply;


-(void)getDirtyBitValue:(void (^)(NSError * _Nullable error,
                                  dirtyBitValue value))reply;

-(bool)isEOFCluster:(uint64_t)cluster;

@end

NS_ASSUME_NONNULL_END

#endif /* FATMANAGER_h */
