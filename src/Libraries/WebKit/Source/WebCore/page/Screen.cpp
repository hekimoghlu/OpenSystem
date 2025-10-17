/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 22, 2025.
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
#include "Screen.h"

#include "Document.h"
#include "DocumentInlines.h"
#include "DocumentLoader.h"
#include "FloatRect.h"
#include "LocalDOMWindow.h"
#include "LocalFrame.h"
#include "LocalFrameView.h"
#include "Page.h"
#include "PlatformScreen.h"
#include "Quirks.h"
#include "ResourceLoadObserver.h"
#include "ScreenOrientation.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(Screen);

Screen::Screen(LocalDOMWindow& window)
    : LocalDOMWindowProperty(&window)
{
}

Screen::~Screen() = default;

static bool shouldApplyScreenFingerprintingProtections(const LocalFrame& frame)
{
    RefPtr page = frame.protectedPage();
    if (!page)
        return false;

    RefPtr document = frame.document();
    if (!document)
        return false;

    return page->shouldApplyScreenFingerprintingProtections(*document);
}

static bool shouldFlipScreenDimensions(const LocalFrame& frame)
{
    RefPtr document = frame.protectedDocument();
    return document && document->quirks().shouldFlipScreenDimensions();
}

int Screen::height() const
{
    RefPtr frame = this->frame();
    if (!frame)
        return 0;
    if (frame->settings().webAPIStatisticsEnabled())
        ResourceLoadObserver::shared().logScreenAPIAccessed(*frame->protectedDocument(), ScreenAPIsAccessed::Height);

    if (shouldFlipScreenDimensions(*frame))
        return static_cast<int>(frame->screenSize().width());

    return static_cast<int>(frame->screenSize().height());
}

int Screen::width() const
{
    RefPtr frame = this->frame();
    if (!frame)
        return 0;
    if (frame->settings().webAPIStatisticsEnabled())
        ResourceLoadObserver::shared().logScreenAPIAccessed(*frame->protectedDocument(), ScreenAPIsAccessed::Width);

    if (shouldFlipScreenDimensions(*frame))
        return static_cast<int>(frame->screenSize().height());

    return static_cast<int>(frame->screenSize().width());
}

unsigned Screen::colorDepth() const
{
    RefPtr frame = this->frame();
    if (!frame)
        return 24;
    if (frame->settings().webAPIStatisticsEnabled())
        ResourceLoadObserver::shared().logScreenAPIAccessed(*frame->protectedDocument(), ScreenAPIsAccessed::ColorDepth);
    return static_cast<unsigned>(screenDepth(frame->protectedView().get()));
}

int Screen::availLeft() const
{
    RefPtr frame = this->frame();
    if (!frame)
        return 0;

    if (frame->settings().webAPIStatisticsEnabled())
        ResourceLoadObserver::shared().logScreenAPIAccessed(*frame->protectedDocument(), ScreenAPIsAccessed::AvailLeft);

    if (shouldApplyScreenFingerprintingProtections(*frame))
        return 0;

    return static_cast<int>(screenAvailableRect(frame->protectedView().get()).x());
}

int Screen::availTop() const
{
    RefPtr frame = this->frame();
    if (!frame)
        return 0;

    if (frame->settings().webAPIStatisticsEnabled())
        ResourceLoadObserver::shared().logScreenAPIAccessed(*frame->protectedDocument(), ScreenAPIsAccessed::AvailTop);

    if (shouldApplyScreenFingerprintingProtections(*frame))
        return 0;

    return static_cast<int>(screenAvailableRect(frame->protectedView().get()).y());
}

int Screen::availHeight() const
{
    RefPtr frame = this->frame();
    if (!frame)
        return 0;

    if (frame->settings().webAPIStatisticsEnabled())
        ResourceLoadObserver::shared().logScreenAPIAccessed(*frame->protectedDocument(), ScreenAPIsAccessed::AvailHeight);

    if (shouldApplyScreenFingerprintingProtections(*frame))
        return static_cast<int>(frame->screenSize().height());

    return static_cast<int>(screenAvailableRect(frame->protectedView().get()).height());
}

int Screen::availWidth() const
{
    RefPtr frame = this->frame();
    if (!frame)
        return 0;

    if (frame->settings().webAPIStatisticsEnabled())
        ResourceLoadObserver::shared().logScreenAPIAccessed(*frame->protectedDocument(), ScreenAPIsAccessed::AvailWidth);

    if (shouldApplyScreenFingerprintingProtections(*frame))
        return static_cast<int>(frame->screenSize().width());

    return static_cast<int>(screenAvailableRect(frame->protectedView().get()).width());
}

ScreenOrientation& Screen::orientation()
{
    if (!m_screenOrientation)
        m_screenOrientation = ScreenOrientation::create(window() ? window()->protectedDocument().get() : nullptr);
    return *m_screenOrientation;
}

} // namespace WebCore
