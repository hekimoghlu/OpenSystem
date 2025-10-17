/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 24, 2025.
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
#import "WebExtensionContext.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#import "WebExtensionAlarm.h"
#import "WebExtensionContextProxyMessages.h"

namespace WebKit {

bool WebExtensionContext::isAlarmsMessageAllowed()
{
    return isLoaded() && hasPermission(WKWebExtensionPermissionAlarms);
}

void WebExtensionContext::alarmsCreate(const String& name, Seconds initialInterval, Seconds repeatInterval)
{
    m_alarmMap.set(name, WebExtensionAlarm::create(name, initialInterval, repeatInterval, [this, protectedThis = Ref { *this }](const WebExtensionAlarm& alarm) {
        fireAlarmsEventIfNeeded(alarm);
    }));
}

void WebExtensionContext::alarmsGet(const String& name, CompletionHandler<void(std::optional<WebExtensionAlarmParameters>&&)>&& completionHandler)
{
    if (RefPtr alarm = m_alarmMap.get(name))
        completionHandler(alarm->parameters());
    else
        completionHandler(std::nullopt);
}

void WebExtensionContext::alarmsClear(const String& name, CompletionHandler<void()>&& completionHandler)
{
    m_alarmMap.remove(name);

    completionHandler();
}

void WebExtensionContext::alarmsGetAll(CompletionHandler<void(Vector<WebExtensionAlarmParameters>&&)>&& completionHandler)
{
    auto alarms = WTF::map(m_alarmMap.values(), [](auto&& alarm) {
        return alarm->parameters();
    });

    completionHandler(WTFMove(alarms));
}

void WebExtensionContext::alarmsClearAll(CompletionHandler<void()>&& completionHandler)
{
    m_alarmMap.clear();

    completionHandler();
}

void WebExtensionContext::fireAlarmsEventIfNeeded(const WebExtensionAlarm& alarm)
{
    constexpr auto type = WebExtensionEventListenerType::AlarmsOnAlarm;
    wakeUpBackgroundContentIfNecessaryToFireEvents({ type }, [=, this, protectedThis = Ref { *this }, alarm = Ref { alarm }] {
        sendToProcessesForEvent(type, Messages::WebExtensionContextProxy::DispatchAlarmsEvent(alarm->parameters()));
    });
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
