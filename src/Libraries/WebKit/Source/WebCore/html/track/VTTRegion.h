/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 30, 2022.
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

#include "ContextDestructionObserver.h"
#include "FloatPoint.h"
#include "TextTrack.h"
#include "Timer.h"

namespace WebCore {

class HTMLDivElement;
class VTTCueBox;
class VTTScanner;

class WEBCORE_EXPORT VTTRegion final : public RefCounted<VTTRegion>, public ContextDestructionObserver {
public:
    static Ref<VTTRegion> create(ScriptExecutionContext& context)
    {
        return adoptRef(*new VTTRegion(context));
    }

    virtual ~VTTRegion();

    const String& id() const { return m_id; }
    void setId(const String&);

    double width() const { return m_width; }
    ExceptionOr<void> setWidth(double);

    unsigned lines() const { return m_lines; }
    void setLines(unsigned);

    double regionAnchorX() const { return m_regionAnchor.x(); }
    ExceptionOr<void> setRegionAnchorX(double);

    double regionAnchorY() const { return m_regionAnchor.y(); }
    ExceptionOr<void> setRegionAnchorY(double);

    double viewportAnchorX() const { return m_viewportAnchor.x(); }
    ExceptionOr<void> setViewportAnchorX(double);

    double viewportAnchorY() const { return m_viewportAnchor.y(); }
    ExceptionOr<void> setViewportAnchorY(double);

    enum class ScrollSetting : bool { EmptyString, Up };
    ScrollSetting scroll() const { return m_scroll; }
    void setScroll(const ScrollSetting);

    void updateParametersFromRegion(const VTTRegion&);

    const String& regionSettings() const { return m_settings; }
    void setRegionSettings(const String&);

    HTMLDivElement& getDisplayTree();
    
    void appendTextTrackCueBox(Ref<TextTrackCueBox>&&);
    void displayLastTextTrackCueBox();
    void willRemoveTextTrackCueBox(VTTCueBox*);

    void cueStyleChanged() { m_recalculateStyles = true; }

private:
    VTTRegion(ScriptExecutionContext&);

    void prepareRegionDisplayTree();

    // The timer is needed to continue processing when cue scrolling ended.
    void startTimer();
    void stopTimer();
    void scrollTimerFired();

    enum RegionSetting {
        None,
        Id,
        Width,
        Lines,
        RegionAnchor,
        ViewportAnchor,
        Scroll
    };

    RegionSetting scanSettingName(VTTScanner&);

    void parseSettingValue(RegionSetting, VTTScanner&);

    static const AtomString& textTrackCueContainerScrollingClass();

    String m_id;
    String m_settings;

    double m_width { 100 };
    unsigned m_lines { 3 };

    FloatPoint m_regionAnchor { 0, 100 };
    FloatPoint m_viewportAnchor { 0, 100 };

    ScrollSetting m_scroll { ScrollSetting::EmptyString };

    // The cue container is the container that is scrolled up to obtain the
    // effect of scrolling cues when this is enabled for the regions.
    RefPtr<HTMLDivElement> m_cueContainer;
    RefPtr<HTMLDivElement> m_regionDisplayTree;

    // Keep track of the current numeric value of the css "top" property.
    double m_currentTop { 0 };

    // The timer is used to display the next cue line after the current one has
    // been displayed. It's main use is for scrolling regions and it triggers as
    // soon as the animation for rolling out one line has finished, but
    // currently it is used also for non-scrolling regions to use a single
    // code path.
    Timer m_scrollTimer;

    bool m_recalculateStyles { true };
};

} // namespace WebCore

#endif
