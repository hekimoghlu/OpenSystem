/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 5, 2023.
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

#if ENABLE(VIDEO)

#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/StringHash.h>

OBJC_CLASS AVMetadataItem;
OBJC_CLASS NSData;
OBJC_CLASS NSDate;
OBJC_CLASS NSDictionary;
OBJC_CLASS NSLocale;
OBJC_CLASS NSNumber;
OBJC_CLASS NSString;

namespace WebCore {

class SerializedPlatformDataCueValue {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SerializedPlatformDataCueValue);
public:
    struct Data {
#if PLATFORM(COCOA)
        String type;
        HashMap<String, String> otherAttributes;
        String key;
        RetainPtr<NSLocale> locale;
        std::variant<std::nullptr_t, RetainPtr<NSString>, RetainPtr<NSDate>, RetainPtr<NSNumber>, RetainPtr<NSData>> value;
        bool operator==(const Data&) const;
#endif
    };

    SerializedPlatformDataCueValue() = default;
    SerializedPlatformDataCueValue(std::optional<Data>&& data)
        : m_data(WTFMove(data)) { }
#if PLATFORM(COCOA)
    SerializedPlatformDataCueValue(AVMetadataItem *);
    RetainPtr<NSDictionary> toNSDictionary() const;
    bool operator==(const SerializedPlatformDataCueValue&) const;
#endif

    const std::optional<Data>& data() const { return m_data; }
private:
    std::optional<Data> m_data;
};

} // namespace WebCore

#endif // ENABLE(VIDEO)
