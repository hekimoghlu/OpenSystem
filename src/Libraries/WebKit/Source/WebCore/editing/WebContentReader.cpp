/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 23, 2025.
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
#include "WebContentReader.h"

#include "Document.h"
#include "DocumentFragment.h"

namespace WebCore {

void WebContentReader::addFragment(Ref<DocumentFragment>&& newFragment)
{
    if (!m_fragment)
        m_fragment = WTFMove(newFragment);
    else
        protectedFragment()->appendChild(newFragment);
}

bool FrameWebContentReader::shouldSanitize() const
{
    RefPtr document = m_frame->document();
    ASSERT(document);
    return document->originIdentifierForPasteboard() != contentOrigin();
}

MSOListQuirks FrameWebContentReader::msoListQuirksForMarkup() const
{
    return contentOrigin().isNull() ? MSOListQuirks::CheckIfNeeded : MSOListQuirks::Disabled;
}

#if PLATFORM(COCOA) || PLATFORM(GTK)
bool WebContentReader::readFilePaths(const Vector<String>& paths)
{
    if (paths.isEmpty() || !frame().document())
        return false;

    for (auto& path : paths)
        readFilePath(path);

    return true;
}
#endif

}

