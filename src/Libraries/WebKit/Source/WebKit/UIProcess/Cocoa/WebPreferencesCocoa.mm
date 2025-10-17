/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 16, 2024.
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
#import "WebPreferences.h"

#import "WebPreferencesKeys.h"
#import <WebCore/RealtimeMediaSourceCenter.h>
#import <wtf/text/MakeString.h>

#if ENABLE(MEDIA_STREAM)
#include "UserMediaPermissionRequestManagerProxy.h"
#endif

namespace WebKit {

static inline NSString *makeKey(const String& identifier, const String& keyPrefix, const String& key)
{
    ASSERT(!identifier.isEmpty());
    return makeString(identifier, keyPrefix, key);
}

bool WebPreferences::platformGetStringUserValueForKey(const String& key, String& userValue)
{
    if (!m_identifier)
        return false;

    NSString *string = [[NSUserDefaults standardUserDefaults] stringForKey:makeKey(m_identifier, m_keyPrefix, key)];
    if (!string)
        return false;

    userValue = string;
    return true;
}

bool WebPreferences::platformGetBoolUserValueForKey(const String& key, bool& userValue)
{
    if (!m_identifier)
        return false;

    id object = [[NSUserDefaults standardUserDefaults] objectForKey:makeKey(m_identifier, m_keyPrefix, key)];
    if (!object)
        return false;
    if (![object respondsToSelector:@selector(boolValue)])
        return false;

    userValue = [object boolValue];
    return true;
}

bool WebPreferences::platformGetUInt32UserValueForKey(const String& key, uint32_t& userValue)
{
    if (!m_identifier)
        return false;

    id object = [[NSUserDefaults standardUserDefaults] objectForKey:makeKey(m_identifier, m_keyPrefix, key)];
    if (!object)
        return false;
    if (![object respondsToSelector:@selector(intValue)])
        return false;

    userValue = [object intValue];
    return true;
}

bool WebPreferences::platformGetDoubleUserValueForKey(const String& key, double& userValue)
{
    if (!m_identifier)
        return false;

    id object = [[NSUserDefaults standardUserDefaults] objectForKey:makeKey(m_identifier, m_keyPrefix, key)];
    if (!object)
        return false;
    if (![object respondsToSelector:@selector(doubleValue)])
        return false;

    userValue = [object doubleValue];
    return true;
}

static id debugUserDefaultsValue(const String& identifier, const String& keyPrefix, const String& globalDebugKeyPrefix, const String& key)
{
    NSUserDefaults *standardUserDefaults = [NSUserDefaults standardUserDefaults];
    id object = nil;

    if (!identifier.isEmpty())
        object = [standardUserDefaults objectForKey:makeKey(identifier, keyPrefix, key)];

    if (!object) {
        // Allow debug preferences to be set globally, using the debug key prefix.
        object = [standardUserDefaults objectForKey:[globalDebugKeyPrefix stringByAppendingString:key]];
    }

    return object;
}

static void setDebugBoolValueIfInUserDefaults(const String& identifier, const String& keyPrefix, const String& globalDebugKeyPrefix, const String& key, WebPreferencesStore& store)
{
    id object = debugUserDefaultsValue(identifier, keyPrefix, globalDebugKeyPrefix, key);
    if (!object)
        return;
    if (![object respondsToSelector:@selector(boolValue)])
        return;

    store.setBoolValueForKey(key, [object boolValue]);
}

static void setDebugUInt32ValueIfInUserDefaults(const String& identifier, const String& keyPrefix, const String& globalDebugKeyPrefix, const String& key, WebPreferencesStore& store)
{
    id object = debugUserDefaultsValue(identifier, keyPrefix, globalDebugKeyPrefix, key);
    if (!object)
        return;
    if (![object respondsToSelector:@selector(unsignedIntegerValue)])
        return;

    store.setUInt32ValueForKey(key, [object unsignedIntegerValue]);
}

void WebPreferences::platformInitializeStore()
{
    @autoreleasepool {
#if ENABLE(MEDIA_STREAM)
        // NOTE: This is set here, and does not setting the default using the 'defaultValue' mechanism, because the
        // 'defaultValue' must be the same in both the UIProcess and WebProcess, which may not be true for audio
        // and video capture state as the WebProcess is not entitled to use the camera or microphone by default.
        // If other preferences need to dynamically set the initial value based on host app state, we should extended
        // the declarative format rather than adding more special cases here.
        m_store.setBoolValueForKey(WebPreferencesKey::mediaDevicesEnabledKey(), UserMediaPermissionRequestManagerProxy::permittedToCaptureAudio() || UserMediaPermissionRequestManagerProxy::permittedToCaptureVideo());
        m_store.setBoolValueForKey(WebPreferencesKey::interruptAudioOnPageVisibilityChangeEnabledKey(),  WebCore::RealtimeMediaSourceCenter::shouldInterruptAudioOnPageVisibilityChange());
#endif

#define INITIALIZE_DEFAULT_OVERRIDABLE_PREFERENCE_FROM_NSUSERDEFAULTS(KeyUpper, KeyLower, TypeName, Type, DefaultValue, HumanReadableName, HumanReadableDescription) \
        setDebug##TypeName##ValueIfInUserDefaults(m_identifier, m_keyPrefix, m_globalDebugKeyPrefix, WebPreferencesKey::KeyLower##Key(), m_store);

        FOR_EACH_DEFAULT_OVERRIDABLE_WEBKIT_PREFERENCE(INITIALIZE_DEFAULT_OVERRIDABLE_PREFERENCE_FROM_NSUSERDEFAULTS)

#undef INITIALIZE_DEFAULT_OVERRIDABLE_PREFERENCE_FROM_NSUSERDEFAULTS

        if (!m_identifier)
            return;

#define INITIALIZE_PREFERENCE_FROM_NSUSERDEFAULTS(KeyUpper, KeyLower, TypeName, Type, DefaultValue, HumanReadableName, HumanReadableDescription) \
        Type user##KeyUpper##Value; \
        if (platformGet##TypeName##UserValueForKey(WebPreferencesKey::KeyLower##Key(), user##KeyUpper##Value)) \
            m_store.set##TypeName##ValueForKey(WebPreferencesKey::KeyLower##Key(), user##KeyUpper##Value);

        FOR_EACH_PERSISTENT_WEBKIT_PREFERENCE(INITIALIZE_PREFERENCE_FROM_NSUSERDEFAULTS)

#undef INITIALIZE_PREFERENCE_FROM_NSUSERDEFAULTS
    }
}

void WebPreferences::platformUpdateStringValueForKey(const String& key, const String& value)
{
    if (!m_identifier)
        return;

    [[NSUserDefaults standardUserDefaults] setObject:value forKey:makeKey(m_identifier, m_keyPrefix, key)];
}

void WebPreferences::platformUpdateBoolValueForKey(const String& key, bool value)
{
    if (!m_identifier)
        return;

    [[NSUserDefaults standardUserDefaults] setBool:value forKey:makeKey(m_identifier, m_keyPrefix, key)];
}

void WebPreferences::platformUpdateUInt32ValueForKey(const String& key, uint32_t value)
{
    if (!m_identifier)
        return;

    [[NSUserDefaults standardUserDefaults] setInteger:value forKey:makeKey(m_identifier, m_keyPrefix, key)];
}

void WebPreferences::platformUpdateDoubleValueForKey(const String& key, double value)
{
    if (!m_identifier)
        return;

    [[NSUserDefaults standardUserDefaults] setDouble:value forKey:makeKey(m_identifier, m_keyPrefix, key)];
}

void WebPreferences::platformUpdateFloatValueForKey(const String& key, float value)
{
    if (!m_identifier)
        return;

    [[NSUserDefaults standardUserDefaults] setFloat:value forKey:makeKey(m_identifier, m_keyPrefix, key)];
}

void WebPreferences::platformDeleteKey(const String& key)
{
    if (!m_identifier)
        return;

    [[NSUserDefaults standardUserDefaults] removeObjectForKey:makeKey(m_identifier, m_keyPrefix, key)];
}

} // namespace WebKit
