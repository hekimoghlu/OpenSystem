/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 4, 2024.
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

#include "FontTaggedSettings.h"
#include <wtf/WeakPtr.h>
#include <wtf/text/AtomString.h>

namespace WebCore {
class FontLoadRequestClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::FontLoadRequestClient> : std::true_type { };
}

namespace WebCore {

class Font;
class FontCreationContext;
class FontDescription;
class FontLoadRequest;
struct FontSelectionSpecifiedCapabilities;

class FontLoadRequestClient : public CanMakeWeakPtr<FontLoadRequestClient> {
public:
    virtual ~FontLoadRequestClient() = default;
    virtual void fontLoaded(FontLoadRequest&) { }
};

class FontLoadRequest {
public:
    virtual ~FontLoadRequest() = default;

    virtual const URL& url() const = 0;
    virtual bool isPending() const = 0;
    virtual bool isLoading() const = 0;
    virtual bool errorOccurred() const = 0;

    virtual bool ensureCustomFontData() = 0;
    virtual RefPtr<Font> createFont(const FontDescription&, bool syntheticBold, bool syntheticItalic, const FontCreationContext&) = 0;

    virtual void setClient(FontLoadRequestClient*) = 0;

    virtual bool isCachedFontLoadRequest() const { return false; }
    virtual bool isWorkerFontLoadRequest() const { return false; }
};

} // namespace WebCore

#define SPECIALIZE_TYPE_TRAITS_FONTLOADREQUEST(ToValueTypeName, predicate) \
SPECIALIZE_TYPE_TRAITS_BEGIN(ToValueTypeName) \
    static bool isType(const WebCore::FontLoadRequest& request) { return request.predicate; } \
SPECIALIZE_TYPE_TRAITS_END()
