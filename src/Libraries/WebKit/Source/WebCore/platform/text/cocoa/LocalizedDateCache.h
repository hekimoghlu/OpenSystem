/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 28, 2024.
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

#include "FontCascade.h"
#include <wtf/HashMap.h>
#include <wtf/RetainPtr.h>

namespace WebCore {
    
enum class DateComponentsType : uint8_t;

class MeasureTextClient {
public:
    virtual float measureText(const String&) const = 0;
    virtual ~MeasureTextClient() = default;
};

class LocalizedDateCache {
public:
    NSDateFormatter *formatterForDateType(DateComponentsType);
    float estimatedMaximumWidthForDateType(DateComponentsType, const FontCascade&, const MeasureTextClient&);
    void localeChanged();

private:
    LocalizedDateCache();
    ~LocalizedDateCache();

    RetainPtr<NSDateFormatter> createFormatterForType(DateComponentsType);
    float estimateMaximumWidth(DateComponentsType, const MeasureTextClient&);

    // Using int instead of DateComponentsType for the key because the enum
    // does not have a default hash and hash traits. Usage of the maps
    // casts the DateComponents::Type into an int as the key.
    typedef UncheckedKeyHashMap<int, RetainPtr<NSDateFormatter>> DateTypeFormatterMap;
    typedef UncheckedKeyHashMap<int, float> DateTypeMaxWidthMap;
    DateTypeFormatterMap m_formatterMap;
    DateTypeMaxWidthMap m_maxWidthMap;
    FontCascade m_font;

    friend LocalizedDateCache& localizedDateCache();
    friend NeverDestroyed<LocalizedDateCache>;
};

// Singleton.
LocalizedDateCache& localizedDateCache();

} // namespace WebCore
