/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 22, 2022.
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
#include "config.h"
#include "WebDateTimePickerMac.h"

#if USE(APPKIT)

#import "AppKitSPI.h"
#import "WebPageProxy.h"

constexpr CGFloat kCalendarWidth = 139;
constexpr CGFloat kCalendarHeight = 148;
constexpr CGFloat kCalendarCornerRadius = 10;
constexpr CGFloat kWindowBorderSize = 0.5;
constexpr NSString * kDateFormatString = @"yyyy-MM-dd";
constexpr NSString * kDateTimeFormatString = @"yyyy-MM-dd'T'HH:mm";
constexpr NSString * kDateTimeWithSecondsFormatString = @"yyyy-MM-dd'T'HH:mm:ss";
constexpr NSString * kDateTimeWithMillisecondsFormatString = @"yyyy-MM-dd'T'HH:mm:ss.SSS";
constexpr NSString * kDefaultLocaleIdentifier = @"en_US_POSIX";
constexpr NSString * kDefaultTimeZoneIdentifier = @"UTC";

@interface WKDateTimePicker : NSObject

- (id)initWithParams:(WebCore::DateTimeChooserParameters&&)params inView:(NSView *)view;
- (void)showPicker:(WebKit::WebDateTimePickerMac&)picker;
- (void)updatePicker:(WebCore::DateTimeChooserParameters&&)params;
- (void)invalidate;

@end

@interface WKDateTimePickerWindow : NSWindow
@end

@interface WKDateTimePickerBackdropView : NSView
@end

namespace WebKit {

Ref<WebDateTimePickerMac> WebDateTimePickerMac::create(WebPageProxy& page, NSView *view)
{
    return adoptRef(*new WebDateTimePickerMac(page, view));
}

WebDateTimePickerMac::~WebDateTimePickerMac()
{
    [m_picker invalidate];
}

WebDateTimePickerMac::WebDateTimePickerMac(WebPageProxy& page, NSView *view)
    : WebDateTimePicker(page)
    , m_view(view)
{
}

void WebDateTimePickerMac::endPicker()
{
    [m_picker invalidate];
    m_picker = nil;
    WebDateTimePicker::endPicker();
}

void WebDateTimePickerMac::showDateTimePicker(WebCore::DateTimeChooserParameters&& params)
{
    if (m_picker) {
        [m_picker updatePicker:WTFMove(params)];
        return;
    }

    m_picker = adoptNS([[WKDateTimePicker alloc] initWithParams:WTFMove(params) inView:m_view.get().get()]);
    [m_picker showPicker:*this];
}

void WebDateTimePickerMac::didChooseDate(StringView date)
{
    if (!m_page)
        return;

    m_page->didChooseDate(date);
}

} // namespace WebKit

@implementation WKDateTimePickerWindow {
    RetainPtr<WKDateTimePickerBackdropView> _backdropView;
}

- (id)initWithContentRect:(NSRect)contentRect styleMask:(NSUInteger)styleMask backing:(NSBackingStoreType)backingStoreType defer:(BOOL)defer
{
    self = [super initWithContentRect:contentRect styleMask:styleMask backing:backingStoreType defer:defer];
    if (!self)
        return nil;

    self.hasShadow = YES;
    self.releasedWhenClosed = NO;
    self.titleVisibility = NSWindowTitleHidden;
    self.titlebarAppearsTransparent = YES;
    self.movable = NO;
    self.backgroundColor = [NSColor clearColor];
    self.opaque = NO;

    _backdropView = adoptNS([[WKDateTimePickerBackdropView alloc] initWithFrame:contentRect]);
    [_backdropView setAutoresizingMask:NSViewWidthSizable | NSViewHeightSizable];
    [self setContentView:_backdropView.get()];

    return self;
}

- (BOOL)canBecomeKeyWindow
{
    return NO;
}

- (BOOL)hasKeyAppearance
{
    return YES;
}

- (NSWindowShadowOptions)shadowOptions
{
    return NSWindowShadowSecondaryWindow;
}

@end

@implementation WKDateTimePickerBackdropView

- (void)drawRect:(NSRect)dirtyRect
{
    [NSGraphicsContext saveGraphicsState];

    [[NSColor controlBackgroundColor] setFill];

    NSRect rect = NSInsetRect(self.frame, kWindowBorderSize, 0);
    NSPoint topLeft = NSMakePoint(NSMinX(rect), NSMaxY(rect));
    NSPoint topRight = NSMakePoint(NSMaxX(rect), NSMaxY(rect));
    NSPoint bottomRight = NSMakePoint(NSMaxX(rect), NSMinY(rect));
    NSPoint bottomLeft = NSMakePoint(NSMinX(rect), NSMinY(rect));

    NSBezierPath *path = [NSBezierPath bezierPath];
    [path moveToPoint:topLeft];
    [path lineToPoint:NSMakePoint(topRight.x - kCalendarCornerRadius, topRight.y)];
    [path curveToPoint:NSMakePoint(topRight.x, topRight.y - kCalendarCornerRadius) controlPoint1:topRight controlPoint2:topRight];
    [path lineToPoint:NSMakePoint(bottomRight.x, bottomRight.y + kCalendarCornerRadius)];
    [path curveToPoint:NSMakePoint(bottomRight.x - kCalendarCornerRadius, bottomRight.y) controlPoint1:bottomRight controlPoint2:bottomRight];
    [path lineToPoint:NSMakePoint(bottomLeft.x + kCalendarCornerRadius, bottomLeft.y)];
    [path curveToPoint:NSMakePoint(bottomLeft.x, bottomLeft.y + kCalendarCornerRadius) controlPoint1:bottomLeft controlPoint2:bottomLeft];
    [path lineToPoint:topLeft];

    [path fill];

    [NSGraphicsContext restoreGraphicsState];
}

@end

@implementation WKDateTimePicker {
    WeakPtr<WebKit::WebDateTimePickerMac> _picker;
    WebCore::DateTimeChooserParameters _params;
    WeakObjCPtr<NSView> _presentingView;

    RetainPtr<WKDateTimePickerWindow> _enclosingWindow;
    RetainPtr<NSDatePicker> _datePicker;
    RetainPtr<NSDateFormatter> _dateFormatter;
}

- (id)initWithParams:(WebCore::DateTimeChooserParameters&&)params inView:(NSView *)view
{
    if (!(self = [super init]))
        return self;

    _presentingView = view;

    NSRect windowRect = [[_presentingView window] convertRectToScreen:[_presentingView convertRect:params.anchorRectInRootView toView:nil]];
    windowRect.origin.y = NSMinY(windowRect) - kCalendarHeight;
    windowRect.size.width = kCalendarWidth;
    windowRect.size.height = kCalendarHeight;

    // Use a UTC timezone as all incoming double values are UTC timestamps. This also ensures that
    // the date value of the NSDatePicker matches the date value returned by JavaScript. The timezone
    // has no effect on the value returned to the WebProcess, as a timezone-agnostic format string is
    // used to return the date.
    NSTimeZone *timeZone = [NSTimeZone timeZoneWithName:kDefaultTimeZoneIdentifier];

    _enclosingWindow = adoptNS([[WKDateTimePickerWindow alloc] initWithContentRect:NSZeroRect styleMask:NSWindowStyleMaskBorderless backing:NSBackingStoreBuffered defer:NO]);
    [_enclosingWindow setFrame:windowRect display:YES];

    _datePicker = adoptNS([[NSDatePicker alloc] initWithFrame:[_enclosingWindow contentView].bounds]);
    [_datePicker setBezeled:NO];
    [_datePicker setDrawsBackground:NO];
    [_datePicker setDatePickerStyle:NSDatePickerStyleClockAndCalendar];
    [_datePicker setDatePickerElements:NSDatePickerElementFlagYearMonthDay];
    [_datePicker setTimeZone:timeZone];
    [_datePicker setTarget:self];
    [_datePicker setAction:@selector(didChooseDate:)];

    auto englishLocale = adoptNS([[NSLocale alloc] initWithLocaleIdentifier:kDefaultLocaleIdentifier]);
    _dateFormatter = adoptNS([[NSDateFormatter alloc] init]);
    [_dateFormatter setLocale:englishLocale.get()];
    [_dateFormatter setTimeZone:timeZone];

    [self updatePicker:WTFMove(params)];

    return self;
}

- (void)showPicker:(WebKit::WebDateTimePickerMac&)picker
{
    _picker = picker;

    [[_enclosingWindow contentView] addSubview:_datePicker.get()];
    [[_presentingView window] addChildWindow:_enclosingWindow.get() ordered:NSWindowAbove];
}

- (void)updatePicker:(WebCore::DateTimeChooserParameters&&)params
{
    _params = WTFMove(params);

    NSString *currentDateValueString = _params.currentValue;

    NSString *format = [self dateFormatStringForType:_params.type];
    [_dateFormatter setDateFormat:format];

    if (![currentDateValueString length])
        [_datePicker setDateValue:[self initialDateForEmptyValue]];
    else {
        NSDate *dateValue = [_dateFormatter dateFromString:currentDateValueString];

        while (!dateValue && (format = [self dateFormatFallbackForFormat:format])) {
            [_dateFormatter setDateFormat:format];
            dateValue = [_dateFormatter dateFromString:currentDateValueString];
        }

        [_datePicker setDateValue:dateValue];
    }

    [_datePicker setMinDate:[NSDate dateWithTimeIntervalSince1970:_params.minimum / 1000.0]];
    [_datePicker setMaxDate:[NSDate dateWithTimeIntervalSince1970:_params.maximum / 1000.0]];

    [_enclosingWindow setAppearance:[NSAppearance appearanceNamed:_params.useDarkAppearance ? NSAppearanceNameDarkAqua : NSAppearanceNameAqua]];
}

- (void)invalidate
{
    [_datePicker removeFromSuperviewWithoutNeedingDisplay];
    [_datePicker setTarget:nil];
    [_datePicker setAction:nil];
    _datePicker = nil;

    _dateFormatter = nil;

    [[_presentingView window] removeChildWindow:_enclosingWindow.get()];
    [_enclosingWindow close];
    _enclosingWindow = nil;
}

- (void)didChooseDate:(id)sender
{
    if (sender != _datePicker)
        return;

    String dateString = [_dateFormatter stringFromDate:[_datePicker dateValue]];
    _picker->didChooseDate(StringView(dateString));
}

- (NSString *)dateFormatStringForType:(NSString *)type
{
    if ([type isEqualToString:@"datetime-local"]) {
        if (_params.hasMillisecondField)
            return kDateTimeWithMillisecondsFormatString;
        if (_params.hasSecondField)
            return kDateTimeWithSecondsFormatString;
        return kDateTimeFormatString;
    }

    return kDateFormatString;
}

- (NSString *)dateFormatFallbackForFormat:(NSString *)format
{
    if ([format isEqualToString:kDateTimeWithMillisecondsFormatString])
        return kDateTimeWithSecondsFormatString;
    if ([format isEqualToString:kDateTimeWithSecondsFormatString])
        return kDateTimeFormatString;

    return nil;
}

- (NSDate *)initialDateForEmptyValue
{
    NSDate *now = [NSDate date];
    NSTimeZone *defaultTimeZone = [NSTimeZone defaultTimeZone];
    NSInteger offset = [defaultTimeZone secondsFromGMTForDate:now];
    return [now dateByAddingTimeInterval:offset];
}

@end

#endif // USE(APPKIT)
