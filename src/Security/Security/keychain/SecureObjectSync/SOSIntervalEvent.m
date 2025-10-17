/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 14, 2024.
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

//
//  SOSIntervalEvent.m
//  Security_ios
//
//

#import <Foundation/Foundation.h>
#import "SOSIntervalEvent.h"
#import "keychain/SecureObjectSync/SOSInternal.h"

/*

 interval setting examples:
 NSTimeInterval earliestGB   = 60*60*24*3;  // wait at least 3 days
 NSTimeInterval latestGB     = 60*60*24*7;  // wait at most 7 days

 pattern:

SOSIntervalEvent fooEvent = [[SOSIntervalEvent alloc] initWithDefaults:account.settings dateDescription:@"foocheck" earliest:60*60*24 latest:60*60*36];

 // should we foo?
    if([fooEvent checkDate]) {
        WeDoFooToo();
        // schedule next foo
        [fooEvent followup];
    }
    // "schedule" is only used if you think there's a date upcoming you don't want altered
    // getDate will return the next schedule event date
 */

@implementation SOSIntervalEvent

- (NSDate *) getDate {
    return [_defaults valueForKey: _dateDescription];
}

- (bool) checkDate {
    NSDate *theDate = [self getDate];
    if(theDate && ([theDate timeIntervalSinceNow] <= 0)) return true;
    return false;
}

- (void) followup {
    NSDate *theDate = SOSCreateRandomDateBetweenNowPlus(_earliestDate, _latestDate);
    [_defaults setValue:theDate forKey: _dateDescription];
}

- (void)schedule {
    NSDate *theDate = [self getDate];
    if(!theDate) {
        [self followup];
    }
}

-(id)initWithDefaults:(NSUserDefaults*) defaults dateDescription:(NSString *)dateDescription earliest:(NSTimeInterval) earliest latest: (NSTimeInterval) latest {
    if ((self = [super init])) {
        _defaults = defaults;
        if(! _defaults) {
            _defaults =  [[NSUserDefaults alloc] init];
        }
        _dateDescription = dateDescription;
        _earliestDate = earliest;
        _latestDate = latest;
        [self schedule];
    }
    return self;
}

@end

