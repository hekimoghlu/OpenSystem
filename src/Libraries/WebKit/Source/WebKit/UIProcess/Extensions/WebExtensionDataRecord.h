/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 13, 2024.
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

#if ENABLE(WK_WEB_EXTENSIONS)

#include "APIObject.h"
#include "WebExtensionDataType.h"
#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

OBJC_CLASS NSArray;
OBJC_CLASS NSMutableArray;
OBJC_CLASS WKWebExtensionDataRecord;

namespace WebKit {

class WebExtensionDataRecord : public API::ObjectImpl<API::Object::Type::WebExtensionDataRecord> {
    WTF_MAKE_NONCOPYABLE(WebExtensionDataRecord);

public:
    template<typename... Args>
    static Ref<WebExtensionDataRecord> create(Args&&... args)
    {
        return adoptRef(*new WebExtensionDataRecord(std::forward<Args>(args)...));
    }

    explicit WebExtensionDataRecord(const String& displayName, const String& uniqueIdentifier);

    using Type = WebExtensionDataType;

    const String& displayName() const { return m_displayName; }
    const String& uniqueIdentifier() const { return m_uniqueIdentifier; }

    OptionSet<Type> types() const;

    size_t totalSize() const;
    size_t sizeOfTypes(OptionSet<Type>) const;

    size_t sizeOfType(Type type) const { return m_typeSizes.get(type); }
    void setSizeOfType(Type type, size_t size) { m_typeSizes.set(type, size); }

#if PLATFORM(COCOA)
    NSArray *errors();
    void addError(NSString *debugDescription, Type);
#endif

#ifdef __OBJC__
    WKWebExtensionDataRecord *wrapper() const { return (WKWebExtensionDataRecord *)API::ObjectImpl<API::Object::Type::WebExtensionDataRecord>::wrapper(); }
#endif

    bool operator==(const WebExtensionDataRecord&) const;

private:
    String m_displayName;
    String m_uniqueIdentifier;
    HashMap<Type, size_t> m_typeSizes;
#if PLATFORM(COCOA)
    RetainPtr<NSMutableArray> m_errors;
#endif
};

class WebExtensionDataRecordHolder : public RefCounted<WebExtensionDataRecordHolder> {
    WTF_MAKE_NONCOPYABLE(WebExtensionDataRecordHolder);
    WTF_MAKE_TZONE_ALLOCATED(WebExtensionDataRecordHolder);

public:
    template<typename... Args>
    static Ref<WebExtensionDataRecordHolder> create(Args&&... args)
    {
        return adoptRef(*new WebExtensionDataRecordHolder(std::forward<Args>(args)...));
    }

    WebExtensionDataRecordHolder() { };

    HashMap<String, Ref<WebExtensionDataRecord>> recordsMap;
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
