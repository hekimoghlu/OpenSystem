/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 11, 2025.
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
#if !__has_feature(objc_arc)
#error This file requires ARC. Add the "-fobjc-arc" compiler flag for this file.
#endif

#import "config.h"
#import "WebExtensionAPIAlarms.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#import "CocoaHelpers.h"
#import "Logging.h"
#import "MessageSenderInlines.h"
#import "WebExtensionAPINamespace.h"
#import "WebExtensionConstants.h"
#import "WebExtensionContextMessages.h"
#import "WebExtensionContextProxy.h"
#import "WebExtensionUtilities.h"
#import "WebProcess.h"
#import <wtf/DateMath.h>

namespace WebKit {

static NSString * const whenKey = @"when";
static NSString * const delayInMinutesKey = @"delayInMinutes";
static NSString * const periodInMinutesKey = @"periodInMinutes";

static NSString * const nameKey = @"name";
static NSString * const scheduledTimeKey = @"scheduledTime";

static NSString * const emptyAlarmName = @"";

static inline NSDictionary *toWebAPI(const WebExtensionAlarmParameters& alarm)
{
    NSMutableDictionary *result = [NSMutableDictionary dictionaryWithCapacity:3];

    result[nameKey] = static_cast<NSString *>(alarm.name);
    result[scheduledTimeKey] = @(floor(alarm.nextScheduledTime.approximateWallTime().secondsSinceEpoch().milliseconds()));

    if (alarm.repeatInterval)
        result[periodInMinutesKey] = @(alarm.repeatInterval.minutes());

    return [result copy];
}

void WebExtensionAPIAlarms::createAlarm(NSString *name, NSDictionary *alarmInfo, NSString **outExceptionString)
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/alarms/create

    static NSDictionary<NSString *, id> *types = @{
        whenKey: NSNumber.class,
        delayInMinutesKey: NSNumber.class,
        periodInMinutesKey: NSNumber.class,
    };

    if (!validateDictionary(alarmInfo, @"info", nil, types, outExceptionString))
        return;

    auto *whenNumber = objectForKey<NSNumber>(alarmInfo, whenKey);
    auto *delayNumber = objectForKey<NSNumber>(alarmInfo, delayInMinutesKey);
    auto *periodNumber = objectForKey<NSNumber>(alarmInfo, periodInMinutesKey);

    if (whenNumber && delayNumber) {
        *outExceptionString = toErrorString(nullString(), @"info", @"it cannot specify both 'delayInMinutes' and 'when'");
        return;
    }

    Seconds when = Seconds::fromMilliseconds(whenNumber.doubleValue);
    Seconds delay = Seconds::fromMinutes(delayNumber.doubleValue);
    Seconds period = Seconds::fromMinutes(periodNumber.doubleValue);
    Seconds currentTime = Seconds::fromMilliseconds(jsCurrentTime());

    Seconds initialInterval;
    Seconds repeatInterval;

    if (when)
        initialInterval = when - currentTime;
    else if (delay)
        initialInterval = delay;

    if (period) {
        repeatInterval = period;

        if (!initialInterval)
            initialInterval = repeatInterval;
    }

    if (!extensionContext().inTestingMode()) {
        // Enforce a minimum interval outside of testing.
        initialInterval = std::max(initialInterval, webExtensionMinimumAlarmInterval);
        repeatInterval = repeatInterval ? std::max(repeatInterval, webExtensionMinimumAlarmInterval) : 0_s;
    }

    WebProcess::singleton().send(Messages::WebExtensionContext::AlarmsCreate(name ?: emptyAlarmName, initialInterval, repeatInterval), extensionContext().identifier());
}

void WebExtensionAPIAlarms::get(NSString *name, Ref<WebExtensionCallbackHandler>&& callback)
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/alarms/get

    WebProcess::singleton().sendWithAsyncReply(Messages::WebExtensionContext::AlarmsGet(name ?: emptyAlarmName), [protectedThis = Ref { *this }, callback = WTFMove(callback)](std::optional<WebExtensionAlarmParameters>&& alarm) {
        callback->call(toWebAPI(alarm));
    }, extensionContext().identifier());
}

void WebExtensionAPIAlarms::getAll(Ref<WebExtensionCallbackHandler>&& callback)
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/alarms/getAll

    WebProcess::singleton().sendWithAsyncReply(Messages::WebExtensionContext::AlarmsGetAll(), [protectedThis = Ref { *this }, callback = WTFMove(callback)](Vector<WebExtensionAlarmParameters> alarms) {
        callback->call(toWebAPI(alarms));
    }, extensionContext().identifier());
}

void WebExtensionAPIAlarms::clear(NSString *name, Ref<WebExtensionCallbackHandler>&& callback)
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/alarms/clear

    WebProcess::singleton().sendWithAsyncReply(Messages::WebExtensionContext::AlarmsClear(name ?: emptyAlarmName), [protectedThis = Ref { *this }, callback = WTFMove(callback)]() {
        callback->call();
    }, extensionContext().identifier());
}

void WebExtensionAPIAlarms::clearAll(Ref<WebExtensionCallbackHandler>&& callback)
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/alarms/clearAll

    WebProcess::singleton().sendWithAsyncReply(Messages::WebExtensionContext::AlarmsClearAll(), [protectedThis = Ref { *this }, callback = WTFMove(callback)]() {
        callback->call();
    }, extensionContext().identifier());
}

WebExtensionAPIEvent& WebExtensionAPIAlarms::onAlarm()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/alarms/onAlarm

    if (!m_onAlarm)
        m_onAlarm = WebExtensionAPIEvent::create(*this, WebExtensionEventListenerType::AlarmsOnAlarm);

    return *m_onAlarm;
}

void WebExtensionContextProxy::dispatchAlarmsEvent(const WebExtensionAlarmParameters& alarm)
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/alarms/onAlarm

    auto *details = toWebAPI(alarm);

    enumerateNamespaceObjects([&](auto& namespaceObject) {
        namespaceObject.alarms().onAlarm().invokeListenersWithArgument(details);
    });
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
