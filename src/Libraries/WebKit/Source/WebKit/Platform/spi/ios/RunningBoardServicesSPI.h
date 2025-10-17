/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 22, 2021.
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

#if USE(APPLE_INTERNAL_SDK)

#import <RunningBoardServices/RunningBoardServices.h>

extern const NSTimeInterval RBSProcessTimeLimitationNone;

#if __has_include(<RunningBoardServices/RBSProcessLimitations.h>)
#import <RunningBoardServices/RBSProcessLimitations.h>
#else
@interface RBSProcessLimitations : NSObject
@property (nonatomic, readwrite, assign) NSTimeInterval runTime;
@end
#endif

#else

#import <mach/message.h>

NS_ASSUME_NONNULL_BEGIN

@interface RBSAttribute : NSObject
@end

@interface RBSDomainAttribute : RBSAttribute
+ (instancetype)attributeWithDomain:(NSString *)domain name:(NSString *)name;
@end

@interface RBSTarget : NSObject
+ (RBSTarget *)targetWithPid:(pid_t)pid;
+ (RBSTarget *)targetWithPid:(pid_t)pid environmentIdentifier:(NSString *)environment;
+ (RBSTarget *)currentProcess;
@end

@class RBSAssertion;
@protocol RBSAssertionObserving;
typedef void (^RBSAssertionInvalidationHandler)(RBSAssertion *assertion, NSError *error);

@interface RBSAssertion : NSObject
- (instancetype)initWithExplanation:(NSString *)explanation target:(RBSTarget *)target attributes:(NSArray <RBSAttribute *> *)attributes;
- (BOOL)acquireWithError:(NSError **)error;
- (void)acquireWithInvalidationHandler:(nullable RBSAssertionInvalidationHandler)handler;
- (void)invalidate;
- (void)addObserver:(id <RBSAssertionObserving>)observer;
- (void)removeObserver:(id <RBSAssertionObserving>)observer;

@property (nonatomic, readonly, assign, getter=isValid) BOOL valid;
@end

@protocol RBSAssertionObserving <NSObject>
- (void)assertionWillInvalidate:(RBSAssertion *)assertion;
- (void)assertion:(RBSAssertion *)assertion didInvalidateWithError:(NSError *)error;
@end

@interface RBSProcessIdentifier : NSObject
+ (RBSProcessIdentifier *)identifierWithPid:(pid_t)pid;
@end

typedef NS_ENUM(uint8_t, RBSTaskState) {
    RBSTaskStateUnknown                 = 0,
    RBSTaskStateNone                    = 1,
    RBSTaskStateRunningUnknown          = 2,
    RBSTaskStateRunningSuspended        = 3,
    RBSTaskStateRunningScheduled        = 4,
};

@interface RBSProcessState : NSObject
@property (nonatomic, readonly, assign) RBSTaskState taskState;
@property (nonatomic, readonly, copy) NSSet<NSString *> *endowmentNamespaces;
@end

extern const NSTimeInterval RBSProcessTimeLimitationNone;

@interface RBSProcessLimitations : NSObject
@property (nonatomic, readwrite, assign) NSTimeInterval runTime;
@end

@interface RBSProcessHandle : NSObject
+ (RBSProcessHandle *)handleForIdentifier:(RBSProcessIdentifier *)identifier error:(NSError **)outError;
+ (RBSProcessHandle *)currentProcess;
@property (nonatomic, readonly, assign) pid_t pid;
@property (nonatomic, readonly, strong) RBSProcessState *currentState;
@property (nonatomic, readonly, strong) RBSProcessLimitations *activeLimitations;
@property (nonatomic, readonly, strong, nullable) RBSProcessHandle *hostProcess;
@property (nonatomic, readonly, assign) audit_token_t auditToken;
@end

@interface RBSProcessStateUpdate : NSObject
@property (nonatomic, readonly, strong) RBSProcessHandle *process;
@property (nonatomic, readonly, strong, nullable) RBSProcessState *state;
@end

@class RBSProcessMonitor;
@class RBSProcessPredicate;
@class RBSProcessStateDescriptor;
@protocol RBSProcessMonitorConfiguring;

typedef void (^RBSProcessMonitorConfigurator)(id<RBSProcessMonitorConfiguring> config);
typedef void (^RBSProcessUpdateHandler)(RBSProcessMonitor *monitor, RBSProcessHandle *process, RBSProcessStateUpdate *update);

@protocol RBSProcessMatching <NSObject>
- (RBSProcessPredicate *)processPredicate;
@end

@protocol RBSProcessMonitorConfiguring
- (void)setPredicates:(nullable NSArray<RBSProcessPredicate *> *)predicates;
- (void)setStateDescriptor:(nullable RBSProcessStateDescriptor *)descriptor;
- (void)setUpdateHandler:(nullable RBSProcessUpdateHandler)block;
@end

@interface RBSProcessMonitor : NSObject <NSCopying>
+ (instancetype)monitorWithConfiguration:(NS_NOESCAPE RBSProcessMonitorConfigurator)block;
- (void)invalidate;
@end

@interface RBSProcessPredicate : NSObject <RBSProcessMatching>
+ (RBSProcessPredicate *)predicateMatchingHandle:(RBSProcessHandle *)process;
typedef NS_OPTIONS(NSUInteger, RBSProcessStateValues) {
    RBSProcessStateValueNone                    = 0,
    RBSProcessStateValueTaskState               = (1 << 0),
    RBSProcessStateValueTags                    = (1 << 1),
    RBSProcessStateValueTerminationResistance   = (1 << 2),
    RBSProcessStateValueLegacyAssertions        = (1 << 3),
    RBSProcessStateValueModernAssertions        = (1 << 4),
};
@end

@interface RBSProcessStateDescriptor : NSObject <NSCopying>
+ (instancetype)descriptor;
@property (nonatomic, readwrite, assign) RBSProcessStateValues values;
@property (nonatomic, readwrite, copy, nullable) NSArray<NSString *> *endowmentNamespaces;
@end

NS_ASSUME_NONNULL_END

#endif
