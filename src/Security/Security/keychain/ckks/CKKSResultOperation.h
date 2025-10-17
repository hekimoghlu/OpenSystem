/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 4, 2023.
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
#if OCTAGON

#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>
#import "keychain/ckks/NSOperationCategories.h"

NS_ASSUME_NONNULL_BEGIN

@class CKKSCondition;

#define CKKSResultErrorDomain @"CKKSResultOperationError"
enum {
    CKKSResultSubresultError = 1,
    CKKSResultSubresultCancelled = 2,
    CKKSResultTimedOut = 3,
};

#define CKKSResultDescriptionErrorDomain @"CKKSResultOperationDescriptionError"

@interface CKKSResultOperation : NSBlockOperation
@property (nullable) NSError* error;
@property (nullable) NSDate* finishDate;
@property CKKSCondition* completionHandlerDidRunCondition;

@property NSInteger descriptionErrorCode; // Set to non-0 for inclusion of this operation in NSError chains. Code is application-dependent, but will be -1 in cases of excessive recursion.
@property (nullable) NSError* descriptionUnderlyingError; // Set to non-nil to include as an underlying error if descriptionErrorCode is used.

// If you subclass CKKSResultOperation, this is the method corresponding to descriptionErrorCode. Fill it in to your heart's content.
- (NSError* _Nullable)descriptionError;

// Very similar to addDependency, but:
//   if the dependent operation has an error or is canceled, cancel this operation
- (void)addSuccessDependency:(CKKSResultOperation*)operation;
- (void)addNullableSuccessDependency:(CKKSResultOperation* _Nullable)operation;

// Call to check if you should run.
// Note: all subclasses must call this if they'd like to comply with addSuccessDependency
// Also sets your .error property to encapsulate the upstream error
- (bool)allDependentsSuccessful;

// Allows you to time out CKKSResultOperations: if they haven't started by now, they'll cancel themselves
// and set their error to indicate the timeout
- (instancetype)timeout:(dispatch_time_t)timeout;

// Convenience constructor.
+ (instancetype)operationWithBlock:(void (^)(void))block;
+ (instancetype)named:(NSString*)name withBlock:(void (^)(void))block NS_SWIFT_NAME(init(name:block:));
+ (instancetype)named:(NSString*)name withBlockTakingSelf:(void(^)(CKKSResultOperation* op))block NS_SWIFT_NAME(init(name:blockTakingSelf:));

// Determine if all these operations were successful, and set this operation's result if not.
- (bool)allSuccessful:(NSArray<CKKSResultOperation*>*)operations;

// Call this to prevent the timeout on this operation from occuring.
// Upon return, either this operation is cancelled, or the timeout will never fire.
- (void)invalidateTimeout;

// Reports the state of this operation. Used for making up description strings.
- (NSString*)operationStateString;
@end

NS_ASSUME_NONNULL_END
#endif // OCTAGON

