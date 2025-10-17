/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 12, 2025.
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
#include "WebInspectorClient.h"

#include <CoreFoundation/CoreFoundation.h>
#include <WebCore/InspectorFrontendClientLocal.h>
#include <WebCore/LocalFrame.h>
#include <WebCore/Page.h>
#include <wtf/RetainPtr.h>
#include <wtf/cf/TypeCastsCF.h>

static constexpr auto inspectorStartsAttachedSetting = "inspectorStartsAttached"_s;
static constexpr auto inspectorAttachDisabledSetting = "inspectorAttachDisabled"_s;

static RetainPtr<CFStringRef> createKeyForPreferences(const String& key)
{
    return adoptCF(CFStringCreateWithFormat(0, 0, CFSTR("WebKit Web Inspector Setting - %@"), key.createCFString().get()));
}

static String loadSetting(const String& key)
{
    auto value = adoptCF(CFPreferencesCopyAppValue(createKeyForPreferences(key).get(), kCFPreferencesCurrentApplication));
    if (auto string = dynamic_cf_cast<CFStringRef>(value.get()))
        return string;
    if (value == kCFBooleanTrue)
        return "true"_s;
    if (value == kCFBooleanFalse)
        return "false"_s;
    return { };
}

static void storeSetting(const String& key, const String& setting)
{
    CFPreferencesSetAppValue(createKeyForPreferences(key).get(), setting.createCFString().get(), kCFPreferencesCurrentApplication);
}

static void deleteSetting(const String& key)
{
    CFPreferencesSetAppValue(createKeyForPreferences(key).get(), nullptr, kCFPreferencesCurrentApplication);
}

void WebInspectorClient::sendMessageToFrontend(const String& message)
{
    m_frontendClient->frontendAPIDispatcher().dispatchMessageAsync(message);
}

bool WebInspectorClient::inspectorAttachDisabled()
{
    return loadSetting(inspectorAttachDisabledSetting) == "true"_s;
}

void WebInspectorClient::setInspectorAttachDisabled(bool disabled)
{
    storeSetting(inspectorAttachDisabledSetting, disabled ? "true"_s : "false"_s);
}

void WebInspectorClient::deleteInspectorStartsAttached()
{
    deleteSetting(inspectorAttachDisabledSetting);
}

bool WebInspectorClient::inspectorStartsAttached()
{
    return loadSetting(inspectorStartsAttachedSetting) == "true"_s;
}

void WebInspectorClient::setInspectorStartsAttached(bool attached)
{
    storeSetting(inspectorStartsAttachedSetting, attached ? "true"_s : "false"_s);
}

void WebInspectorClient::deleteInspectorAttachDisabled()
{
    deleteSetting(inspectorStartsAttachedSetting);
}

std::unique_ptr<WebCore::InspectorFrontendClientLocal::Settings> WebInspectorClient::createFrontendSettings()
{
    class InspectorFrontendSettingsCF : public WebCore::InspectorFrontendClientLocal::Settings {
    private:
        String getProperty(const String& name) final { return loadSetting(name); }
        void setProperty(const String& name, const String& value) final { storeSetting(name, value); }
        void deleteProperty(const String& name) final { deleteSetting(name); }
    };
    return makeUnique<InspectorFrontendSettingsCF>();
}
