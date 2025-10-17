/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 12, 2022.
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
#include "WebPreferencesStore.h"

#include "WebPreferencesKeys.h"
#include <wtf/NeverDestroyed.h>

namespace WebKit {

typedef HashMap<String, bool> BoolOverridesMap;

static BoolOverridesMap& boolTestRunnerOverridesMap()
{
    static NeverDestroyed<BoolOverridesMap> map;
    return map;
}

void WebPreferencesStore::overrideBoolValueForKey(const String& key, bool value)
{
    boolTestRunnerOverridesMap().set(key, value);
}

void WebPreferencesStore::removeTestRunnerOverrides()
{
    boolTestRunnerOverridesMap().clear();
}

template<typename MappedType>
static MappedType valueForKey(const WebPreferencesStore::ValueMap& values, const WebPreferencesStore::ValueMap& overriddenDefaults, const String& key)
{
    auto valuesIt = values.find(key);
    if (valuesIt != values.end() && std::holds_alternative<MappedType>(valuesIt->value))
        return std::get<MappedType>(valuesIt->value);

    auto overriddenDefaultsIt = overriddenDefaults.find(key);
    if (overriddenDefaultsIt != overriddenDefaults.end() && std::holds_alternative<MappedType>(overriddenDefaultsIt->value))
        return std::get<MappedType>(overriddenDefaultsIt->value);

    auto& defaultsMap = WebPreferencesStore::defaults();
    auto defaultsIt = defaultsMap.find(key);
    if (defaultsIt != defaultsMap.end() && std::holds_alternative<MappedType>(defaultsIt->value))
        return std::get<MappedType>(defaultsIt->value);

    return MappedType();
}

template<typename MappedType>
static bool setValueForKey(WebPreferencesStore::ValueMap& map, const WebPreferencesStore::ValueMap& overriddenDefaults, const String& key, const MappedType& value)
{
    MappedType existingValue = valueForKey<MappedType>(map, overriddenDefaults, key);
    if (existingValue == value)
        return false;

    map.set(key, WebPreferencesStore::Value(value));
    return true;
}

bool WebPreferencesStore::setStringValueForKey(const String& key, const String& value)
{
    return setValueForKey<String>(m_values, m_overriddenDefaults, key, value);
}

String WebPreferencesStore::getStringValueForKey(const String& key) const
{
    return valueForKey<String>(m_values, m_overriddenDefaults, key);
}

bool WebPreferencesStore::setBoolValueForKey(const String& key, bool value)
{
    return setValueForKey<bool>(m_values, m_overriddenDefaults, key, value);
}

bool WebPreferencesStore::getBoolValueForKey(const String& key) const
{
    // FIXME: Extend overriding to other key types used from TestRunner.
    auto it = boolTestRunnerOverridesMap().find(key);
    if (it != boolTestRunnerOverridesMap().end())
        return it->value;

    return valueForKey<bool>(m_values, m_overriddenDefaults, key);
}

bool WebPreferencesStore::setUInt32ValueForKey(const String& key, uint32_t value) 
{
    return setValueForKey<uint32_t>(m_values, m_overriddenDefaults, key, value);
}

uint32_t WebPreferencesStore::getUInt32ValueForKey(const String& key) const
{
    return valueForKey<uint32_t>(m_values, m_overriddenDefaults, key);
}

bool WebPreferencesStore::setDoubleValueForKey(const String& key, double value) 
{
    return setValueForKey<double>(m_values, m_overriddenDefaults, key, value);
}

double WebPreferencesStore::getDoubleValueForKey(const String& key) const
{
    return valueForKey<double>(m_values, m_overriddenDefaults, key);
}

// Overridden Defaults

void WebPreferencesStore::setOverrideDefaultsStringValueForKey(const String& key, String value)
{
    m_overriddenDefaults.set(key, Value(value));
}

void WebPreferencesStore::setOverrideDefaultsBoolValueForKey(const String& key, bool value)
{
    m_overriddenDefaults.set(key, Value(value));
}

void WebPreferencesStore::setOverrideDefaultsUInt32ValueForKey(const String& key, uint32_t value)
{
    m_overriddenDefaults.set(key, Value(value));
}

void WebPreferencesStore::setOverrideDefaultsDoubleValueForKey(const String& key, double value)
{
    m_overriddenDefaults.set(key, Value(value));
}

void WebPreferencesStore::deleteKey(const String& key)
{
    m_values.remove(key);
    m_overriddenDefaults.remove(key);
}

} // namespace WebKit
