/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 16, 2025.
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
#include "WebPreferences.h"

#include "APIPageConfiguration.h"
#include "WebPageGroup.h"
#include "WebPageProxy.h"
#include "WebPreferencesKeys.h"
#include "WebProcessPool.h"
#include <WebCore/DeprecatedGlobalSettings.h>
#include <WebCore/LibWebRTCProvider.h>
#include <WebCore/StorageBlockingPolicy.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/ThreadingPrimitives.h>

#if !PLATFORM(COCOA)
#include <WebCore/NotImplemented.h>
#endif

namespace WebKit {

Ref<WebPreferences> WebPreferences::create(const String& identifier, const String& keyPrefix, const String& globalDebugKeyPrefix)
{
    return adoptRef(*new WebPreferences(identifier, keyPrefix, globalDebugKeyPrefix));
}

Ref<WebPreferences> WebPreferences::createWithLegacyDefaults(const String& identifier, const String& keyPrefix, const String& globalDebugKeyPrefix)
{
    auto preferences = WebPreferences::create(identifier, keyPrefix, globalDebugKeyPrefix);
    // FIXME: The registerDefault...ValueForKey machinery is unnecessarily heavyweight and complicated.
    // We can just compute different defaults for modern and legacy APIs in WebPreferencesDefinitions.h macros.
    preferences->registerDefaultUInt32ValueForKey(WebPreferencesKey::storageBlockingPolicyKey(), static_cast<uint32_t>(WebCore::StorageBlockingPolicy::AllowAll));
    return preferences;
}

WebPreferences::WebPreferences(const String& identifier, const String& keyPrefix, const String& globalDebugKeyPrefix)
    : m_identifier(identifier)
    , m_keyPrefix(keyPrefix)
    , m_globalDebugKeyPrefix(globalDebugKeyPrefix)
{
    platformInitializeStore();
}

WebPreferences::WebPreferences(const WebPreferences& other)
    : m_identifier()
    , m_keyPrefix(other.m_keyPrefix)
    , m_globalDebugKeyPrefix(other.m_globalDebugKeyPrefix)
    , m_store(other.m_store)
{
    platformInitializeStore();
}

WebPreferences::~WebPreferences()
{
    ASSERT(m_pages.isEmptyIgnoringNullReferences());
}

const Vector<RefPtr<API::Object>>& WebPreferences::experimentalFeatures()
{
    static auto experimentalFeatures = NeverDestroyed([]() {
        Vector<RefPtr<API::Object>> result;
        for (auto& object : features()) {
            API::FeatureStatus status = static_pointer_cast<API::Feature>(object)->status();
            if (status == API::FeatureStatus::Developer || status == API::FeatureStatus::Testable || status == API::FeatureStatus::Preview || status == API::FeatureStatus::Stable)
                result.append(object);
        }
        return result;
    }());
    return experimentalFeatures;
}

const Vector<RefPtr<API::Object>>& WebPreferences::internalDebugFeatures()
{
    static auto internalDebugFeatures = NeverDestroyed([]() {
        Vector<RefPtr<API::Object>> result;
        for (auto& object : features()) {
            API::FeatureStatus status = static_pointer_cast<API::Feature>(object)->status();
            if (status == API::FeatureStatus::Unstable || status == API::FeatureStatus::Internal)
                result.append(object);
        }
        return result;
    }());
    return internalDebugFeatures;
}

Ref<WebPreferences> WebPreferences::copy() const
{
    return adoptRef(*new WebPreferences(*this));
}

void WebPreferences::addPage(WebPageProxy& webPageProxy)
{
    ASSERT(!m_pages.contains(webPageProxy));
    m_pages.add(webPageProxy);
}

void WebPreferences::removePage(WebPageProxy& webPageProxy)
{
    ASSERT(m_pages.contains(webPageProxy));
    m_pages.remove(webPageProxy);
}

void WebPreferences::update()
{
    if (m_updateBatchCount) {
        m_needUpdateAfterBatch = true;
        return;
    }
        
    for (auto& webPageProxy : m_pages)
        webPageProxy.preferencesDidChange();
}

void WebPreferences::startBatchingUpdates()
{
    if (!m_updateBatchCount)
        m_needUpdateAfterBatch = false;

    ++m_updateBatchCount;
}

void WebPreferences::endBatchingUpdates()
{
    ASSERT(m_updateBatchCount > 0);
    --m_updateBatchCount;
    if (!m_updateBatchCount && m_needUpdateAfterBatch)
        update();
}

void WebPreferences::setBoolValueForKey(const String& key, bool value, bool ephemeral)
{
    if (!m_store.setBoolValueForKey(key, value))
        return;
    updateBoolValueForKey(key, value, ephemeral);
}

void WebPreferences::setDoubleValueForKey(const String& key, double value, bool ephemeral)
{
    if (!m_store.setDoubleValueForKey(key, value))
        return;
    updateDoubleValueForKey(key, value, ephemeral);
}

void WebPreferences::setUInt32ValueForKey(const String& key, uint32_t value, bool ephemeral)
{
    if (!m_store.setUInt32ValueForKey(key, value))
        return;
    updateUInt32ValueForKey(key, value, ephemeral);
}

void WebPreferences::setStringValueForKey(const String& key, const String& value, bool ephemeral)
{
    if (!m_store.setStringValueForKey(key, value))
        return;
    updateStringValueForKey(key, value, ephemeral);
}

void WebPreferences::updateStringValueForKey(const String& key, const String& value, bool ephemeral)
{
    platformUpdateStringValueForKey(key, value);
    update(); // FIXME: Only send over the changed key and value.
}

void WebPreferences::updateBoolValueForKey(const String& key, bool value, bool ephemeral)
{
    if (!ephemeral)
        platformUpdateBoolValueForKey(key, value);
    
    if (key == WebPreferencesKey::processSwapOnCrossSiteNavigationEnabledKey()) {
        for (auto& page : m_pages)
            page.configuration().processPool().configuration().setProcessSwapsOnNavigation(value);

        return;
    }

    update(); // FIXME: Only send over the changed key and value.
}

void WebPreferences::updateUInt32ValueForKey(const String& key, uint32_t value, bool ephemeral)
{
    platformUpdateUInt32ValueForKey(key, value);
    update(); // FIXME: Only send over the changed key and value.
}

void WebPreferences::updateDoubleValueForKey(const String& key, double value, bool ephemeral)
{
    platformUpdateDoubleValueForKey(key, value);
    update(); // FIXME: Only send over the changed key and value.
}

void WebPreferences::updateFloatValueForKey(const String& key, float value, bool ephemeral)
{
    platformUpdateFloatValueForKey(key, value);
    update(); // FIXME: Only send over the changed key and value.
}

void WebPreferences::deleteKey(const String& key)
{
    m_store.deleteKey(key);
    platformDeleteKey(key);
    update(); // FIXME: Only send over the changed key and value.
}

void WebPreferences::registerDefaultBoolValueForKey(const String& key, bool value)
{
    m_store.setOverrideDefaultsBoolValueForKey(key, value);
    bool userValue;
    if (platformGetBoolUserValueForKey(key, userValue))
        m_store.setBoolValueForKey(key, userValue);
}

void WebPreferences::registerDefaultUInt32ValueForKey(const String& key, uint32_t value)
{
    m_store.setOverrideDefaultsUInt32ValueForKey(key, value);
    uint32_t userValue;
    if (platformGetUInt32UserValueForKey(key, userValue))
        m_store.setUInt32ValueForKey(key, userValue);
}

#if !PLATFORM(COCOA) && !PLATFORM(GTK) && !PLATFORM(WPE)
void WebPreferences::platformInitializeStore()
{
    notImplemented();
}
#endif

#if !PLATFORM(COCOA)
void WebPreferences::platformUpdateStringValueForKey(const String&, const String&)
{
    notImplemented();
}

void WebPreferences::platformUpdateBoolValueForKey(const String&, bool)
{
    notImplemented();
}

void WebPreferences::platformUpdateUInt32ValueForKey(const String&, uint32_t)
{
    notImplemented();
}

void WebPreferences::platformUpdateDoubleValueForKey(const String&, double)
{
    notImplemented();
}

void WebPreferences::platformUpdateFloatValueForKey(const String&, float)
{
    notImplemented();
}

void WebPreferences::platformDeleteKey(const String&)
{
    notImplemented();
}

bool WebPreferences::platformGetStringUserValueForKey(const String&, String&)
{
    notImplemented();
    return false;
}

bool WebPreferences::platformGetBoolUserValueForKey(const String&, bool&)
{
    notImplemented();
    return false;
}

bool WebPreferences::platformGetUInt32UserValueForKey(const String&, uint32_t&)
{
    notImplemented();
    return false;
}

bool WebPreferences::platformGetDoubleUserValueForKey(const String&, double&)
{
    notImplemented();
    return false;
}
#endif

} // namespace WebKit
