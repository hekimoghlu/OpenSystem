/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 24, 2023.
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
#if __OBJC2__

#import "SFAnalyticsActivityTracker.h"
#import "SFAnalyticsActivityTracker+Internal.h"
#import "SFAnalytics.h"
#import <mach/mach_time.h>
#import "utilities/debugging.h"

@interface SFAnalyticsActivityTracker ()
@property (readwrite) NSNumber * measurement;
@end

@implementation SFAnalyticsActivityTracker {
    dispatch_queue_t _queue;
    NSString* _name;
    Class _clientClass;
    uint64_t _start;
    BOOL _canceled;
}

@synthesize measurement = _measurement;

- (instancetype)initWithName:(NSString*)name clientClass:(Class)className {
    if (![name isKindOfClass:[NSString class]] || ![className isSubclassOfClass:[SFAnalytics class]] ) {
        secerror("Cannot instantiate SFActivityTracker without name and client class");
        return nil;
    }

    if (self = [super init]) {
        _queue = dispatch_queue_create("SFAnalyticsActivityTracker queue", DISPATCH_QUEUE_SERIAL_WITH_AUTORELEASE_POOL);
        _name = name;
        _clientClass = className;
        _measurement = nil;
        _canceled = NO;
        _start = 0;
    }
    return self;
}

- (void)performAction:(void (^)(void))action
{
    [self start];
    dispatch_sync(_queue, ^{
        action();
    });
    [self stop];
}

- (void)start
{
    if (_canceled) {
        return;
    }
    NSAssert(_start == 0, @"SFAnalyticsActivityTracker user called start twice");
    _start = mach_absolute_time();
}

- (void)stop
{
    uint64_t end = mach_absolute_time();

    if (_canceled) {
        _start = 0;
        return;
    }
    NSAssert(_start != 0, @"SFAnalyticsActivityTracker user called stop w/o calling start");
    
    static mach_timebase_info_data_t sTimebaseInfo;
    if ( sTimebaseInfo.denom == 0 ) {
        (void)mach_timebase_info(&sTimebaseInfo);
    }

    _measurement = @([_measurement doubleValue] + (1.0f * (end - _start) * (1.0f * sTimebaseInfo.numer / sTimebaseInfo.denom)));
    _start = 0;
}

- (void)stopWithEvent:(NSString*)eventName
               result:(NSError* _Nullable)eventResultError
{
    [self stop];

    [[_clientClass logger] logResultForEvent:eventName hardFailure:false result:eventResultError];
}

- (void)cancel
{
    _canceled = YES;
}

- (void)dealloc
{
    if (_start != 0) {
        [self stop];
    }
    if (!_canceled && _measurement != nil) {
        [[_clientClass logger] logMetric:_measurement withName:_name];
    }
}

@end

#endif
