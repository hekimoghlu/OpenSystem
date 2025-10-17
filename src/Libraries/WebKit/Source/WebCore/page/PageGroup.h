/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 27, 2023.
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

#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakHashSet.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class PageGroup;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::PageGroup> : std::true_type { };
}

namespace WebCore {

class Page;
#if ENABLE(VIDEO)
class CaptionUserPreferences;
#endif

class PageGroup : public CanMakeWeakPtr<PageGroup> {
    WTF_MAKE_TZONE_ALLOCATED(PageGroup);
    WTF_MAKE_NONCOPYABLE(PageGroup);
public:
    WEBCORE_EXPORT explicit PageGroup(const String& name);
    explicit PageGroup(Page&);
    ~PageGroup();

    WEBCORE_EXPORT static PageGroup* pageGroup(const String& groupName);

    const WeakHashSet<Page>& pages() const { return m_pages; }

    void addPage(Page&);
    void removePage(Page&);

    const String& name() { return m_name; }
    unsigned identifier() { return m_identifier; }

#if ENABLE(VIDEO)
    WEBCORE_EXPORT void captionPreferencesChanged();
    WEBCORE_EXPORT CaptionUserPreferences& ensureCaptionPreferences();
    CaptionUserPreferences* captionPreferences() const { return m_captionPreferences.get(); }
#endif

private:
    String m_name;
    WeakHashSet<Page> m_pages;

    unsigned m_identifier;

#if ENABLE(VIDEO)
    RefPtr<CaptionUserPreferences> m_captionPreferences;
#endif
};

} // namespace WebCore
