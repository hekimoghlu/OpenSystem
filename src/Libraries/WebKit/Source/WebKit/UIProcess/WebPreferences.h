/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 15, 2025.
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
#pragma once

#include "APIFeature.h"
#include "APIObject.h"
#include "WebPreferencesDefinitions.h"
#include "WebPreferencesStore.h"
#include <wtf/RefPtr.h>
#include <wtf/WeakHashSet.h>

#define DECLARE_PREFERENCE_GETTER_AND_SETTERS(KeyUpper, KeyLower, TypeName, Type, DefaultValue, HumanReadableName, HumanReadableDescription) \
    void set##KeyUpper(const Type& value); \
    void delete##KeyUpper(); \
    Type KeyLower() const;

#define DECLARE_INSPECTOR_OVERRIDE_SETTERS(KeyUpper, KeyLower, Type) \
    void set##KeyUpper##InspectorOverride(std::optional<Type> inspectorOverride);

#define DECLARE_INSPECTOR_OVERRIDE_STORE(KeyUpper, KeyLower, Type) \
    std::optional<Type> m_##KeyLower##InspectorOverride;


namespace WebKit {

class WebPageProxy;

class WebPreferences : public API::ObjectImpl<API::Object::Type::Preferences> {
public:
    static Ref<WebPreferences> create(const String& identifier, const String& keyPrefix, const String& globalDebugKeyPrefix);
    static Ref<WebPreferences> createWithLegacyDefaults(const String& identifier, const String& keyPrefix, const String& globalDebugKeyPrefix);

    explicit WebPreferences(const String& identifier, const String& keyPrefix, const String& globalDebugKeyPrefix);
    WebPreferences(const WebPreferences&);

    virtual ~WebPreferences();

    Ref<WebPreferences> copy() const;

    void addPage(WebPageProxy&);
    void removePage(WebPageProxy&);

    const WebPreferencesStore& store() const { return m_store; }

    // Implemented in generated file WebPreferencesGetterSetters.cpp.
    FOR_EACH_WEBKIT_PREFERENCE(DECLARE_PREFERENCE_GETTER_AND_SETTERS)
    FOR_EACH_WEBKIT_PREFERENCE_WITH_INSPECTOR_OVERRIDE(DECLARE_INSPECTOR_OVERRIDE_SETTERS)

    static const Vector<RefPtr<API::Object>>& features();
    static const Vector<RefPtr<API::Object>>& experimentalFeatures();
    static const Vector<RefPtr<API::Object>>& internalDebugFeatures();
    
    bool isFeatureEnabled(const API::Feature&) const;
    void setFeatureEnabled(const API::Feature&, bool);
    void setFeatureEnabledForKey(const String&, bool);

    // FIXME: Update for unified feature semantics
    // enableAllExperimentalFeatures() should enable settings for testing based on status, or be replaced with an API that WebKitTestRunner can use to enable arbitrary settings.
    void enableAllExperimentalFeatures();
    void resetAllInternalDebugFeatures();
    void disableRichJavaScriptFeatures();
    void disableMediaPlaybackRelatedFeatures();

    // Exposed for WebKitTestRunner use only.
    void setBoolValueForKey(const String&, bool value, bool ephemeral);
    void setDoubleValueForKey(const String&, double value, bool ephemeral);
    void setUInt32ValueForKey(const String&, uint32_t value, bool ephemeral);
    void setStringValueForKey(const String&, const String& value, bool ephemeral);
    void forceUpdate() { update(); }

    void startBatchingUpdates();
    void endBatchingUpdates();

private:
    void platformInitializeStore();

    void update();

    class UpdateBatch {
    public:
        explicit UpdateBatch(WebPreferences& preferences)
            : m_preferences(preferences)
        {
            m_preferences->startBatchingUpdates();
        }
        
        ~UpdateBatch()
        {
            m_preferences->endBatchingUpdates();
        }
        
    private:
        Ref<WebPreferences> m_preferences;
    };

    void updateStringValueForKey(const String& key, const String& value, bool ephemeral);
    void updateBoolValueForKey(const String& key, bool value, bool ephemeral);
    void updateUInt32ValueForKey(const String& key, uint32_t value, bool ephemeral);
    void updateDoubleValueForKey(const String& key, double value, bool ephemeral);
    void updateFloatValueForKey(const String& key, float value, bool ephemeral);
    void platformUpdateStringValueForKey(const String& key, const String& value);
    void platformUpdateBoolValueForKey(const String& key, bool value);
    void platformUpdateUInt32ValueForKey(const String& key, uint32_t value);
    void platformUpdateDoubleValueForKey(const String& key, double value);
    void platformUpdateFloatValueForKey(const String& key, float value);

    void deleteKey(const String& key);
    void platformDeleteKey(const String& key);

    void registerDefaultBoolValueForKey(const String&, bool);
    void registerDefaultUInt32ValueForKey(const String&, uint32_t);

    bool platformGetStringUserValueForKey(const String& key, String& userValue);
    bool platformGetBoolUserValueForKey(const String&, bool&);
    bool platformGetUInt32UserValueForKey(const String&, uint32_t&);
    bool platformGetDoubleUserValueForKey(const String&, double&);

    const String m_identifier;
    const String m_keyPrefix;
    const String m_globalDebugKeyPrefix;
    WebPreferencesStore m_store;

    WeakHashSet<WebPageProxy> m_pages;
    unsigned m_updateBatchCount { 0 };
    bool m_needUpdateAfterBatch { false };

    FOR_EACH_WEBKIT_PREFERENCE_WITH_INSPECTOR_OVERRIDE(DECLARE_INSPECTOR_OVERRIDE_STORE)
};

} // namespace WebKit
