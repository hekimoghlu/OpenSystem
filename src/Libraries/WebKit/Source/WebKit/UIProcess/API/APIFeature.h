/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 24, 2022.
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

#include "APIFeatureStatus.h"
#include "APIObject.h"
#include <wtf/text/WTFString.h>

namespace API {

class Feature final : public ObjectImpl<Object::Type::Feature> {
public:

    template <FeatureStatus Status, bool DefaultValue>
    static Ref<Feature> create(const WTF::String& name, const WTF::String& key, FeatureConstant<Status> status, FeatureCategory category, const WTF::String& details, std::bool_constant<DefaultValue> defaultValue, bool hidden)
    {
#if ENABLE(FEATURE_DEFAULT_VALIDATION)
        constexpr auto impliedDefaultValue = API::defaultValueForFeatureStatus(Status);
        if constexpr (impliedDefaultValue && *impliedDefaultValue)
            static_assert(defaultValue, "Feature's status implies it should be on by default");
        else if constexpr (impliedDefaultValue && !*impliedDefaultValue)
            static_assert(!defaultValue, "Feature's status implies it should be off by default");
#endif

        return uncheckedCreate(name, key, status, category, details, defaultValue, hidden);
    }

    template <FeatureStatus Status>
    static Ref<Feature> create(const WTF::String& name, const WTF::String& key, FeatureConstant<Status> status, FeatureCategory category, const WTF::String& details, bool defaultValue, bool hidden)
    {
        return uncheckedCreate(name, key, status, category, details, defaultValue, hidden);
    }

    virtual ~Feature() = default;

    WTF::String name() const { return m_name; }
    WTF::String key() const { return m_key; }
    FeatureStatus status() const { return m_status; }
    FeatureCategory category() const { return m_category; }
    WTF::String details() const { return m_details; }
    bool defaultValue() const { return m_defaultValue; }
    bool isHidden() const { return m_hidden; }
    
private:
    explicit Feature(const WTF::String& name, const WTF::String& key, FeatureStatus, FeatureCategory, const WTF::String& details, bool defaultValue, bool hidden);

    static Ref<Feature> uncheckedCreate(const WTF::String& name, const WTF::String& key, FeatureStatus, FeatureCategory, const WTF::String& details, bool defaultValue, bool hidden);

    WTF::String m_name;
    WTF::String m_key;
    WTF::String m_details;
    FeatureStatus m_status;
    FeatureCategory m_category;
    bool m_defaultValue;
    bool m_hidden;
};

}
