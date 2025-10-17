/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 2, 2022.
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
#import "config.h"
#import "Regress141275.h"

#import <Foundation/Foundation.h>
#import <objc/objc.h>
#import <objc/runtime.h>

#if JSC_OBJC_API_ENABLED

extern "C" void JSSynchronousGarbageCollectForDebugging(JSContextRef);

extern int failed;

static const NSUInteger scriptToEvaluate = 50;

@interface JSTEvaluator : NSObject
- (instancetype)initWithScript:(NSString*)script;

- (void)insertSignPostWithCompletion:(void(^)(NSError* error))completionHandler;

- (void)evaluateScript:(NSString*)script completion:(void(^)(NSError* error))completionHandler;
- (void)evaluateBlock:(void(^)(JSContext* context))evaluationBlock completion:(void(^)(NSError* error))completionHandler;

- (void)waitForTasksDoneAndReportResults;
@end


static const NSString* JSTEvaluatorThreadContextKey = @"JSTEvaluatorThreadContextKey";

/*
 * A JSTEvaluatorThreadContext is kept in the thread dictionary of threads used by JSEvaluator.
 *
 * This includes the run loop thread, and any threads used by _jsSourcePerformQueue to execute a task.
 */
@interface JSTEvaluatorThreadContext : NSObject
@property (weak) JSTEvaluator* evaluator;
@property (strong) JSContext* jsContext;
@end

@implementation JSTEvaluatorThreadContext
@end


/*!
 * A JSTEvaluatorTask is a single task to be executed.
 *
 * JSTEvaluator keeps a list of pending tasks. The run loop thread is repsonsible for feeding pending tasks to the _jsSourcePerformQueue, while respecting sign posts.
 */
@interface JSTEvaluatorTask : NSObject

@property (nonatomic, copy) void (^evaluateBlock)(JSContext* jsContext);
@property (nonatomic, copy) void (^completionHandler)(NSError* error);
@property (nonatomic, copy) NSError* error;

+ (instancetype)evaluatorTaskWithEvaluateBlock:(void (^)(JSContext*))block completionHandler:(void (^)(NSError* error))completionBlock;

@end

@implementation JSTEvaluatorTask

+ (instancetype)evaluatorTaskWithEvaluateBlock:(void (^)(JSContext*))evaluationBlock completionHandler:(void (^)(NSError* error))completionHandler
{
    JSTEvaluatorTask* task = [self new];
    task.evaluateBlock = evaluationBlock;
    task.completionHandler = completionHandler;
    return task;
}

@end

@implementation JSTEvaluator {
    dispatch_queue_t _jsSourcePerformQueue;
    dispatch_semaphore_t _allScriptsDone;
    CFRunLoopRef _jsThreadRunLoop;
    CFRunLoopSourceRef _jsThreadRunLoopSource;
    JSContext* _jsContext;
    NSMutableArray* __pendingTasks;
}

- (instancetype)init
{
    self = [super init];
    if (self) {
        _jsSourcePerformQueue = dispatch_queue_create("JSTEval", DISPATCH_QUEUE_CONCURRENT);

        _allScriptsDone = dispatch_semaphore_create(0);

        _jsContext = [JSContext new];
        _jsContext.name = @"JSTEval";
        __pendingTasks = [NSMutableArray new];

        NSThread* jsThread = [[NSThread alloc] initWithTarget:self selector:@selector(_jsThreadMain) object:nil];
        [jsThread setName:@"JSTEval"];
        [jsThread start];

    }
    return self;
}

- (instancetype)initWithScript:(NSString*)script
{
    self = [self init];
    if (self) {
        dispatch_semaphore_t dsema = dispatch_semaphore_create(0);
        [self evaluateScript:script
            completion:^(NSError*) {
                dispatch_semaphore_signal(dsema);
            }];
        dispatch_semaphore_wait(dsema, DISPATCH_TIME_FOREVER);
    }
    return self;
}

- (void)_accessPendingTasksWithBlock:(void(^)(NSMutableArray* pendingTasks))block
{
    @synchronized(self) {
        block(__pendingTasks);
        if (__pendingTasks.count > 0) {
            if (_jsThreadRunLoop && _jsThreadRunLoopSource) {
                CFRunLoopSourceSignal(_jsThreadRunLoopSource);
                CFRunLoopWakeUp(_jsThreadRunLoop);
            }
        }
    }
}

- (void)insertSignPostWithCompletion:(void(^)(NSError* error))completionHandler
{
    [self _accessPendingTasksWithBlock:^(NSMutableArray* pendingTasks) {
        JSTEvaluatorTask* task = [JSTEvaluatorTask evaluatorTaskWithEvaluateBlock:nil
            completionHandler:completionHandler];

        [pendingTasks addObject:task];
    }];
}

- (void)evaluateScript:(NSString*)script completion:(void(^)(NSError* error))completionHandler
{
    [self evaluateBlock:^(JSContext* context) {
        [context evaluateScript:script];
    } completion:completionHandler];
}

- (void)evaluateBlock:(void(^)(JSContext* context))evaluationBlock completion:(void(^)(NSError* error))completionHandler
{
    NSParameterAssert(evaluationBlock != nil);
    [self _accessPendingTasksWithBlock:^(NSMutableArray* pendingTasks) {
        JSTEvaluatorTask* task = [JSTEvaluatorTask evaluatorTaskWithEvaluateBlock:evaluationBlock
            completionHandler:completionHandler];

        [pendingTasks addObject:task];
    }];
}

- (void)waitForTasksDoneAndReportResults
{
    NSString* passFailString = @"PASSED";

    if (!dispatch_semaphore_wait(_allScriptsDone, dispatch_time(DISPATCH_TIME_NOW, 30 * NSEC_PER_SEC))) {
        int totalScriptsRun = [_jsContext[@"counter"] toInt32];

        if (totalScriptsRun != scriptToEvaluate) {
            passFailString = @"FAILED";
            failed = 1;
        }

        NSLog(@"  Ran a total of %d scripts: %@", totalScriptsRun, passFailString);
    } else {
        passFailString = @"FAILED";
        failed = 1;
        NSLog(@"  Error, timeout waiting for all tasks to complete: %@", passFailString);
    }
}

static void __JSTRunLoopSourceScheduleCallBack(void* info, CFRunLoopRef rl, CFStringRef)
{
    @autoreleasepool {
        [(__bridge JSTEvaluator*)info _sourceScheduledOnRunLoop:rl];
    }
}

static void __JSTRunLoopSourcePerformCallBack(void* info )
{
    @autoreleasepool {
        [(__bridge JSTEvaluator*)info _sourcePerform];
    }
}

static void __JSTRunLoopSourceCancelCallBack(void* info, CFRunLoopRef rl, CFStringRef)
{
    @autoreleasepool {
        [(__bridge JSTEvaluator*)info _sourceCanceledOnRunLoop:rl];
    }
}

- (void)_jsThreadMain
{
    @autoreleasepool {
        const CFIndex kRunLoopSourceContextVersion = 0;
        CFRunLoopSourceContext sourceContext = {
            kRunLoopSourceContextVersion, (__bridge void*)(self),
            NULL, NULL, NULL, NULL, NULL,
            __JSTRunLoopSourceScheduleCallBack,
            __JSTRunLoopSourceCancelCallBack,
            __JSTRunLoopSourcePerformCallBack
        };

        @synchronized(self) {
            _jsThreadRunLoop = CFRunLoopGetCurrent();
            CFRetain(_jsThreadRunLoop);

            _jsThreadRunLoopSource = CFRunLoopSourceCreate(kCFAllocatorDefault, 0, &sourceContext);
            CFRunLoopAddSource(_jsThreadRunLoop, _jsThreadRunLoopSource, kCFRunLoopDefaultMode);
        }

        CFRunLoopRun();

        @synchronized(self) {
            NSMutableDictionary* threadDict = [[NSThread currentThread] threadDictionary];
            [threadDict removeObjectForKey:threadDict[JSTEvaluatorThreadContextKey]];

            CFRelease(_jsThreadRunLoopSource);
            _jsThreadRunLoopSource = NULL;

            CFRelease(_jsThreadRunLoop);
            _jsThreadRunLoop = NULL;

            __pendingTasks = nil;
        }
    }
}

- (void)_sourceScheduledOnRunLoop:(CFRunLoopRef)runLoop
{
    UNUSED_PARAM(runLoop);
    assert([[[NSThread currentThread] name] isEqualToString:@"JSTEval"]);

    // Wake up the run loop in case requests were submitted prior to the
    // run loop & run loop source getting created.
    CFRunLoopSourceSignal(_jsThreadRunLoopSource);
    CFRunLoopWakeUp(_jsThreadRunLoop);
}

- (void)_setupEvaluatorThreadContextIfNeeded
{
    NSMutableDictionary* threadDict = [[NSThread currentThread] threadDictionary];
    JSTEvaluatorThreadContext* context = threadDict[JSTEvaluatorThreadContextKey];
    // The evaluator may be other evualuator, or nil if this thread has not been used before. Eaither way take ownership.
    if (context.evaluator != self) {
        context = [JSTEvaluatorThreadContext new];
        context.evaluator = self;
        threadDict[JSTEvaluatorThreadContextKey] = context;
    }
}

- (void)_callCompletionHandler:(void(^)(NSError* error))completionHandler ifNeededWithError:(NSError*)error
{
    if (completionHandler) {
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
            completionHandler(error);
        });
    }
}

- (void)_sourcePerform
{
    assert([[[NSThread currentThread] name] isEqualToString:@"JSTEval"]);

    __block NSArray* tasks = nil;
    [self _accessPendingTasksWithBlock:^(NSMutableArray* pendingTasks) {
        // No signpost, take all tasks.
        tasks = [pendingTasks copy];
        [pendingTasks removeAllObjects];
    }];

    if (tasks.count > 0) {
        for (JSTEvaluatorTask* task in tasks) {
            dispatch_block_t block = ^{
                NSError* error = nil;
                if (task.evaluateBlock) {
                    [self _setupEvaluatorThreadContextIfNeeded];
                    task.evaluateBlock(self->_jsContext);
                    if (self->_jsContext.exception) {
                        NSLog(@"Did fail on JSContext: %@", self->_jsContext.name);
                        NSDictionary* userInfo = @{ NSLocalizedDescriptionKey : [self->_jsContext.exception[@"message"] toString] };
                        error = [NSError errorWithDomain:@"JSTEvaluator" code:1 userInfo:userInfo];
                        self->_jsContext.exception = nil;
                    }
                }
                [self _callCompletionHandler:task.completionHandler ifNeededWithError:error];
            };

            if (task.evaluateBlock)
                dispatch_async(_jsSourcePerformQueue, block);
            else
                dispatch_barrier_async(_jsSourcePerformQueue, block);
        }

        dispatch_barrier_sync(_jsSourcePerformQueue, ^{
            if ([self->_jsContext[@"counter"] toInt32] == scriptToEvaluate)
                dispatch_semaphore_signal(self->_allScriptsDone);
        });
    }
}

- (void)_sourceCanceledOnRunLoop:(CFRunLoopRef)runLoop
{
    UNUSED_PARAM(runLoop);
    assert([[[NSThread currentThread] name] isEqualToString:@"JSTEval"]);

    @synchronized(self) {
        assert(_jsThreadRunLoop);
        assert(_jsThreadRunLoopSource);

        CFRunLoopRemoveSource(_jsThreadRunLoop, _jsThreadRunLoopSource, kCFRunLoopDefaultMode);
        CFRunLoopStop(_jsThreadRunLoop);
    }
}

@end

void runRegress141275()
{
    // Test that we can execute the same script from multiple threads with a shared context.
    // See <https://webkit.org/b/141275>
    NSLog(@"TEST: Testing multiple threads executing the same script with a shared context");

    @autoreleasepool {
        JSTEvaluator* evaluator = [[JSTEvaluator alloc] initWithScript:@"this['counter'] = 0;"];

        void (^showErrorIfNeeded)(NSError* error) = ^(NSError* error) {
            if (error) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    NSLog(@"Error: %@", error);
                });
            }
        };

        [evaluator evaluateBlock:^(JSContext* context) {
            JSSynchronousGarbageCollectForDebugging([context JSGlobalContextRef]);
        } completion:showErrorIfNeeded];

        [evaluator evaluateBlock:^(JSContext* context) {
            context[@"wait"] = ^{
                [NSThread sleepForTimeInterval:0.01];
            };
        } completion:^(NSError* error) {
            if (error) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    NSLog(@"Error: %@", error);
                });
            }
            for (unsigned i = 0; i < scriptToEvaluate; i++)
                [evaluator evaluateScript:@"this['counter']++; this['wait']();" completion:showErrorIfNeeded];
        }];

        [evaluator waitForTasksDoneAndReportResults];
    }
}

#endif // JSC_OBJC_API_ENABLED
