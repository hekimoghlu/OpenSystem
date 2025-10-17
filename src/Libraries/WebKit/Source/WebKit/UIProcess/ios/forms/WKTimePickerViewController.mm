/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 20, 2022.
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
#import "WKTimePickerViewController.h"

#if HAVE(PEPPER_UI_CORE)

#import "ClockKitSPI.h"
#import "UIKitSPI.h"

#import <WebCore/LocalizedStrings.h>
#import <wtf/SoftLinking.h>
#import <wtf/text/WTFString.h>

SOFT_LINK_PRIVATE_FRAMEWORK(ClockKitUI)
SOFT_LINK_CLASS(ClockKitUI, CLKUIWheelsOfTimeView)

static NSString *timePickerTimeZoneForFormatting = @"UTC";
static NSString *timePickerDateFormat = @"HH:mm";

@interface WKTimePickerViewController () <CLKUIWheelsOfTimeDelegate>
@end

@implementation WKTimePickerViewController {
    RetainPtr<CLKUIWheelsOfTimeView> _timePicker;
    RetainPtr<NSDateFormatter> _dateFormatter;
}

@dynamic delegate;

- (instancetype)initWithDelegate:(id <WKQuickboardViewControllerDelegate>)delegate
{
    self = [super initWithDelegate:delegate];
    return self;
}

- (NSDateFormatter *)dateFormatter
{
    if (_dateFormatter)
        return _dateFormatter.get();

    _dateFormatter = adoptNS([[NSDateFormatter alloc] init]);
    [_dateFormatter setLocale:[NSLocale localeWithLocaleIdentifier:@"en_US_POSIX"]];
    [_dateFormatter setDateFormat:timePickerDateFormat];
    [_dateFormatter setTimeZone:[NSTimeZone timeZoneWithName:timePickerTimeZoneForFormatting]];
    return _dateFormatter.get();
}

- (NSString *)timeValueForFormControls
{
    NSCalendar *calendar = [NSCalendar calendarWithIdentifier:NSCalendarIdentifierGregorian];
    calendar.timeZone = [NSTimeZone timeZoneWithName:timePickerTimeZoneForFormatting];

    NSDate *epochDate = [NSDate dateWithTimeIntervalSince1970:0];
    NSDate *timePickerDateAsOffsetFromEpoch = [calendar dateBySettingHour:[_timePicker hour] minute:[_timePicker minute] second:0 ofDate:epochDate options:0];
    return [self.dateFormatter stringFromDate:timePickerDateAsOffsetFromEpoch];
}

- (NSDateComponents *)dateComponentsFromInitialValue
{
    NSString *initialText = [self.delegate initialValueForViewController:self];
    if (initialText.length < timePickerDateFormat.length)
        return nil;

    NSString *truncatedInitialValue = [initialText substringToIndex:timePickerDateFormat.length];
    NSDate *parsedDate = [self.dateFormatter dateFromString:truncatedInitialValue];
    if (!parsedDate)
        return nil;

    NSCalendar *calendar = [NSCalendar calendarWithIdentifier:NSCalendarIdentifierGregorian];
    calendar.timeZone = [NSTimeZone timeZoneWithName:timePickerTimeZoneForFormatting];
    return [calendar components:NSCalendarUnitHour | NSCalendarUnitMinute fromDate:parsedDate];
}

#pragma mark - UIViewController overrides

- (void)viewDidAppear:(BOOL)animated
{
    [super viewDidAppear:animated];
    [self becomeFirstResponder];
}

- (void)viewDidLoad
{
    [super viewDidLoad];

    self.view.backgroundColor = UIColor.systemBackgroundColor;

    self.headerView.hidden = YES;

    _timePicker = adoptNS([allocCLKUIWheelsOfTimeViewInstance() initWithFrame:self.view.bounds style:CLKUIWheelsOfTimeStyleAlarm12]);
    [_timePicker setDelegate:self];

    NSDateComponents *components = self.dateComponentsFromInitialValue;
    if (components)
        [_timePicker setHour:components.hour andMinute:components.minute];

    [self.contentView addSubview:_timePicker.get()];
}

- (BOOL)becomeFirstResponder
{
    return [_timePicker becomeFirstResponder];
}

- (void)setHour:(NSInteger)hour minute:(NSInteger)minute
{
    [_timePicker setHour:hour andMinute:minute];
    [self rightButtonWOTAction];
}

#pragma mark - CLKUIWheelsOfTimeDelegate

- (void)leftButtonWOTAction
{
    // Handle an action on the 'Cancel' button.
    ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    [self.delegate quickboardInputCancelled:static_cast<id<PUICQuickboardController>>(self)];
    ALLOW_DEPRECATED_DECLARATIONS_END
}

- (void)rightButtonWOTAction
{
    // Handle an action on the 'Set' button.
    auto valueAsAttributedString = adoptNS([[NSAttributedString alloc] initWithString:self.timeValueForFormControls]);
    ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    [self.delegate quickboard:static_cast<id<PUICQuickboardController>>(self) textEntered:valueAsAttributedString.get()];
    ALLOW_DEPRECATED_DECLARATIONS_END
}

@end

#endif // HAVE(PEPPER_UI_CORE)
