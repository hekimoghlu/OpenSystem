/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 10, 2023.
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

#include "Decoder.h"
#include "Encoder.h"
#include <wtf/CheckedRef.h>
#include <wtf/CrossThreadCopier.h>
#include <wtf/RobinHoodHashMap.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

// FIXME: WebPreferencesStore should be RefCounted. See usage in WebProcessPool.cpp.

struct WebPreferencesStore {
    using Value = std::variant<String, bool, uint32_t, double>;
    using ValueMap = MemoryCompactRobinHoodHashMap<String, Value>;

    // NOTE: The getters in this class have non-standard names to aid in the use of the preference macros.

    bool setStringValueForKey(const String& key, const String& value);
    String getStringValueForKey(const String& key) const;

    bool setBoolValueForKey(const String& key, bool value);
    bool getBoolValueForKey(const String& key) const;

    bool setUInt32ValueForKey(const String& key, uint32_t value);
    uint32_t getUInt32ValueForKey(const String& key) const;

    bool setDoubleValueForKey(const String& key, double value);
    double getDoubleValueForKey(const String& key) const;

    void setOverrideDefaultsStringValueForKey(const String& key, String value);
    void setOverrideDefaultsBoolValueForKey(const String& key, bool value);
    void setOverrideDefaultsUInt32ValueForKey(const String& key, uint32_t value);
    void setOverrideDefaultsDoubleValueForKey(const String& key, double value);

    void deleteKey(const String& key);

    // For WebKitTestRunner usage.
    static void overrideBoolValueForKey(const String& key, bool value);
    static void removeTestRunnerOverrides();

    ValueMap m_values { };
    ValueMap m_overriddenDefaults { };

    WebPreferencesStore isolatedCopy() const & { return { crossThreadCopy(m_values), crossThreadCopy(m_overriddenDefaults) }; }
    WebPreferencesStore isolatedCopy() && { return { crossThreadCopy(WTFMove(m_values)), crossThreadCopy(WTFMove(m_overriddenDefaults)) }; }

    static ValueMap& defaults();
};

} // namespace WebKit
