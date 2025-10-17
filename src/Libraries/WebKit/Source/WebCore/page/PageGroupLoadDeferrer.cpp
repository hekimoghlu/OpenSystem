/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 16, 2023.
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
#include "PageGroupLoadDeferrer.h"

#include "Document.h"
#include "DocumentParser.h"
#include "LocalFrame.h"
#include "Page.h"
#include "PageGroup.h"
#include "ScriptRunner.h"

namespace WebCore {

PageGroupLoadDeferrer::PageGroupLoadDeferrer(Page& page, bool deferSelf)
{
    for (auto& otherPage : page.group().pages()) {
        if (!deferSelf && &otherPage == &page)
            continue;
        if (otherPage.defersLoading())
            continue;
        auto* localMainFrame = dynamicDowncast<LocalFrame>(otherPage.mainFrame());
        if (!localMainFrame)
            continue;
        m_deferredFrames.append(localMainFrame);

        // This code is not logically part of load deferring, but we do not want JS code executed beneath modal
        // windows or sheets, which is exactly when PageGroupLoadDeferrer is used.
        for (RefPtr<Frame> frame = localMainFrame; frame; frame = frame->tree().traverseNext()) {
            RefPtr localFrame = dynamicDowncast<LocalFrame>(frame);
            if (!localFrame)
                continue;
            localFrame->protectedDocument()->suspendScheduledTasks(ReasonForSuspension::WillDeferLoading);
        }
    }

    for (auto& deferredFrame : m_deferredFrames) {
        if (Page* page = deferredFrame->page())
            page->setDefersLoading(true);
    }
}

PageGroupLoadDeferrer::~PageGroupLoadDeferrer()
{
    for (auto& deferredFrame : m_deferredFrames) {
        auto* page = deferredFrame->page();
        if (!page)
            continue;
        page->setDefersLoading(false);

        for (RefPtr frame = &page->mainFrame(); frame; frame = frame->tree().traverseNext()) {
            RefPtr localFrame = dynamicDowncast<LocalFrame>(frame);
            if (!localFrame)
                continue;
            localFrame->protectedDocument()->resumeScheduledTasks(ReasonForSuspension::WillDeferLoading);
        }
    }
}


} // namespace WebCore
