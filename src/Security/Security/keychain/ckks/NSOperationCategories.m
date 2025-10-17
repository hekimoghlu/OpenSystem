/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 18, 2024.
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
#import "keychain/ckks/CKKS.h"
#import "keychain/ckks/NSOperationCategories.h"
#import "keychain/ot/ObjCImprovements.h"

#define LINEAR_DEPENDENCY_LIMIT 100

@implementation NSOperation (CKKSUsefulPrintingOperation)
- (NSString*)selfname {
    if(self.name) {
        return [NSString stringWithFormat: @"%@(%@)", NSStringFromClass([self class]), self.name];
    } else {
        return NSStringFromClass([self class]);
    }
}

-(void)linearDependencies: (NSHashTable*) collection {
    @synchronized(collection) {
        for(NSOperation* existingop in collection) {
            if(existingop == self) {
                // don't depend on yourself
                continue;
            }
#if DEBUG
            ckksnotice_global("ckks-operation", "adding dependency of %@ on %@", self.name, existingop);
#endif
            [self addDependency: existingop];
        }
        [collection addObject:self];
        if (collection.count > LINEAR_DEPENDENCY_LIMIT) {
            secerror("ckks-operation: linear dependencies exceeds %d operations", LINEAR_DEPENDENCY_LIMIT);
        }
    }
}

-(void)linearDependenciesWithSelfFirst: (NSHashTable*) collection {
    @synchronized(collection) {
        for(NSOperation* existingop in collection) {
            if(existingop == self) {
                // don't depend on yourself
                continue;
            }

            if([existingop isPending]) {
                [existingop addDependency: self];
                if([existingop isPending]) {
                    // Good, we're ahead of this one.
                } else {
                    // It started before we told it to wait on us. Reverse the dependency.
                    [existingop removeDependency: self];
                    [self addDependency:existingop];
                }
            } else {
                // Not a pending op? We depend on it.
                [self addDependency: existingop];
            }
        }
        [collection addObject:self];
    }
}

-(NSString*)pendingDependenciesString:(NSString*)prefix {
    static __thread unsigned __descriptionRecursion = 0;

    @autoreleasepool {
        NSArray* dependencies = [self.dependencies copy];
        dependencies = [dependencies objectsAtIndexes: [dependencies indexesOfObjectsPassingTest: ^BOOL (id obj,
                                                                                                         NSUInteger idx,
                                                                                                         BOOL* stop) {
            return [obj isFinished] ? NO : YES;
        }]];

        if(dependencies.count == 0u) {
            return @"";
        }

        if(__descriptionRecursion > 20) {
            return [NSString stringWithFormat: @"<recursion too deep>"];
        }
        __descriptionRecursion++;

        // Clear the autoreleasepool for each individual depedency.
        NSMutableArray<NSString*>* dependenciesStrings = [NSMutableArray array];
        for(NSOperation* dep in dependencies) {
            @autoreleasepool {
                [dependenciesStrings addObject:[dep description]];
            }
        }

        __descriptionRecursion--;

        return [NSString stringWithFormat: @"%@%@", prefix, [dependenciesStrings componentsJoinedByString: @", "]];
    }
}

- (NSString*)description {
    NSString* state = ([self isFinished] ? @"finished" :
                       [self isCancelled] ? @"cancelled" :
                       [self isExecuting] ? @"executing" :
                       [self isReady] ? @"ready" :
                       @"pending");

    return [NSString stringWithFormat: @"<%@: %@%@>", [self selfname], state, [self pendingDependenciesString: @" dep:"]];
}
- (NSString*)debugDescription {
    NSString* state = ([self isFinished] ? @"finished" :
                       [self isCancelled] ? @"cancelled" :
                       [self isExecuting] ? @"executing" :
                       [self isReady] ? @"ready" :
                       @"pending");

    return [NSString stringWithFormat: @"<%@ (%p): %@%@>", [self selfname], self, state, [self pendingDependenciesString: @" dep:"]];
}

- (BOOL)isPending {
    return (!([self isExecuting] || [self isFinished] || [self isCancelled])) ? YES : NO;
}

- (void)addNullableDependency: (NSOperation*) op {
    if(op) {
        [self addDependency:op];
    }
}

- (void)removeDependenciesUponCompletion
{
    WEAKIFY(self);
    self.completionBlock = ^{
        STRONGIFY(self);
        for (NSOperation *op in self.dependencies) {
            [self removeDependency:op];
        }
    };
}

@end

@implementation NSBlockOperation (CKKSUsefulConstructorOperation)
+(instancetype)named: (NSString*)name withBlock: (void(^)(void)) block {
    // How many blocks could a block block if a block could block blocks?
    NSBlockOperation* blockOp = [NSBlockOperation blockOperationWithBlock: block];
    blockOp.name = name;
    return blockOp;
}
@end
