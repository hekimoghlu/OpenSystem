/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 19, 2021.
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
#import "DateTests.h"
#import <Foundation/Foundation.h>

#if JSC_OBJC_API_ENABLED

extern "C" void checkResult(NSString *description, bool passed);

@interface DateTests : NSObject
+ (void) NSDateToJSDateTest;
+ (void) JSDateToNSDateTest;
+ (void) roundTripThroughJSDateTest;
+ (void) roundTripThroughObjCDateTest;
@end

static unsigned unitFlags = NSCalendarUnitSecond | NSCalendarUnitMinute | NSCalendarUnitHour | NSCalendarUnitDay | NSCalendarUnitMonth | NSCalendarUnitYear;

@implementation DateTests
+ (void) NSDateToJSDateTest
{
    JSContext *context = [[JSContext alloc] init];
    NSDate *now = [NSDate dateWithTimeIntervalSinceNow:0];
    NSDateComponents *components = [[NSCalendar currentCalendar] components:unitFlags fromDate:now];
    JSValue *jsNow = [JSValue valueWithObject:now inContext:context];
    int year = [[jsNow invokeMethod:@"getFullYear" withArguments:@[]] toInt32];
    // Months are 0-indexed for JavaScript Dates.
    int month = [[jsNow invokeMethod:@"getMonth" withArguments:@[]] toInt32] + 1;
    int day = [[jsNow invokeMethod:@"getDate" withArguments:@[]] toInt32];
    int hour = [[jsNow invokeMethod:@"getHours" withArguments:@[]] toInt32];
    int minute = [[jsNow invokeMethod:@"getMinutes" withArguments:@[]] toInt32];
    int second = [[jsNow invokeMethod:@"getSeconds" withArguments:@[]] toInt32];

    checkResult(@"NSDate to JS Date", year == [components year]
        && month == [components month]
        && day == [components day]
        && hour == [components hour]
        && minute == [components minute]
        && second == [components second]);
}

+ (void) JSDateToNSDateTest
{
    JSContext *context = [[JSContext alloc] init];
    NSDateFormatter *formatter = [[NSDateFormatter alloc] init];
    [formatter setDateFormat:@"MMMM dd',' yyyy hh:mm:ss"];
    [formatter setLocale:[[NSLocale alloc] initWithLocaleIdentifier:@"en_US"]];
    NSDate *februaryFourth2014 = [formatter dateFromString:@"February 4, 2014 11:40:03"];
    NSDateComponents *components = [[NSCalendar currentCalendar] components:unitFlags fromDate:februaryFourth2014];
    // Months are 0-indexed for JavaScript Dates.
    JSValue *jsDate = [context[@"Date"] constructWithArguments:@[@2014, @1, @4, @11, @40, @3]];
    
    int year = [[jsDate invokeMethod:@"getFullYear" withArguments:@[]] toInt32];
    int month = [[jsDate invokeMethod:@"getMonth" withArguments:@[]] toInt32] + 1;
    int day = [[jsDate invokeMethod:@"getDate" withArguments:@[]] toInt32];
    int hour = [[jsDate invokeMethod:@"getHours" withArguments:@[]] toInt32];
    int minute = [[jsDate invokeMethod:@"getMinutes" withArguments:@[]] toInt32];
    int second = [[jsDate invokeMethod:@"getSeconds" withArguments:@[]] toInt32];

    checkResult(@"JS Date to NSDate", year == [components year]
        && month == [components month]
        && day == [components day]
        && hour == [components hour]
        && minute == [components minute]
        && second == [components second]);
}

+ (void) roundTripThroughJSDateTest
{
    JSContext *context = [[JSContext alloc] init];
    [context evaluateScript:@"function jsReturnDate(date) { return date; }"];
    NSDateFormatter *formatter = [[NSDateFormatter alloc] init];
    [formatter setDateFormat:@"MMMM dd',' yyyy hh:mm:ss"];
    [formatter setLocale:[[NSLocale alloc] initWithLocaleIdentifier:@"en_US"]];
    NSDate *februaryFourth2014 = [formatter dateFromString:@"February 4, 2014 11:40:03"];
    NSDateComponents *components = [[NSCalendar currentCalendar] components:unitFlags fromDate:februaryFourth2014];
    
    JSValue *roundTripThroughJS = [context[@"jsReturnDate"] callWithArguments:@[februaryFourth2014]];
    int year = [[roundTripThroughJS invokeMethod:@"getFullYear" withArguments:@[]] toInt32];
    // Months are 0-indexed for JavaScript Dates.
    int month = [[roundTripThroughJS invokeMethod:@"getMonth" withArguments:@[]] toInt32] + 1;
    int day = [[roundTripThroughJS invokeMethod:@"getDate" withArguments:@[]] toInt32];
    int hour = [[roundTripThroughJS invokeMethod:@"getHours" withArguments:@[]] toInt32];
    int minute = [[roundTripThroughJS invokeMethod:@"getMinutes" withArguments:@[]] toInt32];
    int second = [[roundTripThroughJS invokeMethod:@"getSeconds" withArguments:@[]] toInt32];

    checkResult(@"JS date round trip", year == [components year]
        && month == [components month]
        && day == [components day]
        && hour == [components hour]
        && minute == [components minute]
        && second == [components second]);
}

+ (void) roundTripThroughObjCDateTest
{
    JSContext *context = [[JSContext alloc] init];
    context[@"objcReturnDate"] = ^(NSDate *date) {
        return date;
    };
    [context evaluateScript:@"function test() {\
        var date = new Date(2014, 1, 4, 11, 40, 3); \
        var result = objcReturnDate(date); \
        return date.getYear() === result.getYear() \
            && date.getMonth() === result.getMonth() \
            && date.getDate() === result.getDate() \
            && date.getHours() === result.getHours() \
            && date.getMinutes() === result.getMinutes() \
            && date.getSeconds() === result.getSeconds() \
            && date.getMilliseconds() === result.getMilliseconds();\
    }"];
    
    checkResult(@"ObjC date round trip", [[context[@"test"] callWithArguments:@[]] toBool]);
}

@end

void runDateTests()
{
    @autoreleasepool {
        [DateTests NSDateToJSDateTest];
        [DateTests JSDateToNSDateTest];
        [DateTests roundTripThroughJSDateTest];
        [DateTests roundTripThroughObjCDateTest];
    }
}

#endif // JSC_OBJC_API_ENABLED
