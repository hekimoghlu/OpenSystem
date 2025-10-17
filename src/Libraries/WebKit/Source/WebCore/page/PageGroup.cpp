/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 29, 2024.
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
#include "PageGroup.h"

#include "BackForwardCache.h"
#include "DOMWrapperWorld.h"
#include "Document.h"
#include "LocalFrame.h"
#include "Page.h"
#include "StorageNamespace.h"
#include <JavaScriptCore/HeapInlines.h>
#include <JavaScriptCore/StructureInlines.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(VIDEO)
#if PLATFORM(MAC) || HAVE(MEDIA_ACCESSIBILITY_FRAMEWORK)
#include "CaptionUserPreferencesMediaAF.h"
#else
#include "CaptionUserPreferences.h"
#endif
#endif

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(PageGroup);

static unsigned getUniqueIdentifier()
{
    static unsigned currentIdentifier = 0;
    return ++currentIdentifier;
}

// --------

PageGroup::PageGroup(const String& name)
    : m_name(name)
    , m_identifier(getUniqueIdentifier())
{
}

PageGroup::PageGroup(Page& page)
    : m_identifier(getUniqueIdentifier())
{
    addPage(page);
}

PageGroup::~PageGroup() = default;

using PageGroupMap = HashMap<String, PageGroup*>;
static PageGroupMap* pageGroups = nullptr;

PageGroup* PageGroup::pageGroup(const String& groupName)
{
    ASSERT(!groupName.isEmpty());
    
    if (!pageGroups)
        pageGroups = new PageGroupMap;

    PageGroupMap::AddResult result = pageGroups->add(groupName, nullptr);

    if (result.isNewEntry) {
        ASSERT(!result.iterator->value);
        result.iterator->value = new PageGroup(groupName);
    }

    ASSERT(result.iterator->value);
    return result.iterator->value;
}

void PageGroup::addPage(Page& page)
{
    ASSERT(!m_pages.contains(page));
    m_pages.add(page);
}

void PageGroup::removePage(Page& page)
{
    ASSERT(m_pages.contains(page));
    m_pages.remove(page);
}

#if ENABLE(VIDEO)
void PageGroup::captionPreferencesChanged()
{
    for (auto& page : m_pages)
        page.captionPreferencesChanged();
    BackForwardCache::singleton().markPagesForCaptionPreferencesChanged();
}

CaptionUserPreferences& PageGroup::ensureCaptionPreferences()
{
    if (!m_captionPreferences) {
#if PLATFORM(MAC) || HAVE(MEDIA_ACCESSIBILITY_FRAMEWORK)
        m_captionPreferences = CaptionUserPreferencesMediaAF::create(*this);
#else
        m_captionPreferences = CaptionUserPreferences::create(*this);
#endif
    }

    return *m_captionPreferences.get();
}
#endif

} // namespace WebCore
