/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 20, 2025.
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

#include "ExceptionOr.h"
#include "MediaQuery.h"
#include "MediaQueryParserContext.h"
#include <memory>
#include <wtf/Forward.h>
#include <wtf/Vector.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

class CSSRule;
class CSSStyleSheet;

class MediaList final : public RefCounted<MediaList> {
public:
    static Ref<MediaList> create(CSSStyleSheet* parentSheet)
    {
        return adoptRef(*new MediaList(parentSheet));
    }
    static Ref<MediaList> create(CSSRule* parentRule)
    {
        return adoptRef(*new MediaList(parentRule));
    }

    WEBCORE_EXPORT ~MediaList();

    bool isSupportedPropertyIndex(unsigned index) const { return index < length(); }
    WEBCORE_EXPORT unsigned length() const;
    WEBCORE_EXPORT String item(unsigned index) const;
    WEBCORE_EXPORT ExceptionOr<void> deleteMedium(const String& oldMedium);
    WEBCORE_EXPORT void appendMedium(const String& newMedium);

    WEBCORE_EXPORT String mediaText() const;
    WEBCORE_EXPORT void setMediaText(const String&);

    CSSRule* parentRule() const { return m_parentRule; }
    CSSStyleSheet* parentStyleSheet() const { return m_parentStyleSheet; }
    void detachFromParent();

    const MQ::MediaQueryList& mediaQueries() const;

private:
    MediaList(CSSStyleSheet* parentSheet);
    MediaList(CSSRule* parentRule);

    void setMediaQueries(MQ::MediaQueryList&&);

    CSSStyleSheet* m_parentStyleSheet { nullptr };
    CSSRule* m_parentRule { nullptr };
    std::optional<MQ::MediaQueryList> m_detachedMediaQueries;
};

WTF::TextStream& operator<<(WTF::TextStream&, const MediaList&);

} // namespace
