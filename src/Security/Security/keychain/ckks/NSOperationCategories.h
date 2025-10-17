/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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
NS_ASSUME_NONNULL_BEGIN

@interface NSOperation (CKKSUsefulPrintingOperation)
- (NSString*)description;
- (BOOL)isPending;

// Use our .name field if it's there, otherwise, just a generic name
- (NSString*)selfname;

// If op is nonnull, op becomes a dependency of this operation
- (void)addNullableDependency:(NSOperation* _Nullable)op;

// Add all operations in this collection as dependencies, then add yourself to the collection
- (void)linearDependencies:(NSHashTable*)collection;

// Insert yourself as high up the linearized list of dependencies as possible
- (void)linearDependenciesWithSelfFirst:(NSHashTable*)collection;

// Set completionBlock to remove all dependencies - break strong references.
- (void)removeDependenciesUponCompletion;

// Return a stringified representation of this operation's live dependencies.
- (NSString*)pendingDependenciesString:(NSString*)prefix;
@end

@interface NSBlockOperation (CKKSUsefulConstructorOperation)
+ (instancetype)named:(NSString*)name withBlock:(void (^)(void))block;
@end

NS_ASSUME_NONNULL_END
