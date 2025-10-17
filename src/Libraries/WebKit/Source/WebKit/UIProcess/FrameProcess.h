/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 26, 2024.
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

#include <WebCore/Site.h>
#include <wtf/Ref.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>

namespace WebKit {

class BrowsingContextGroup;
class WebPreferences;
class WebProcessProxy;

// Note: This object should only be referenced by WebFrameProxy because its destructor is an
// important part of managing the lifetime of a frame and the process used by the frame.
class FrameProcess : public RefCountedAndCanMakeWeakPtr<FrameProcess> {
public:
    ~FrameProcess();

    const WebCore::Site& site() const { return m_site; }
    const WebProcessProxy& process() const { return m_process.get(); }
    WebProcessProxy& process() { return m_process.get(); }

private:
    friend class BrowsingContextGroup; // FrameProcess should not be created except by BrowsingContextGroup.
    static Ref<FrameProcess> create(WebProcessProxy& process, BrowsingContextGroup& group, const WebCore::Site& site, const WebPreferences& preferences) { return adoptRef(*new FrameProcess(process, group, site, preferences)); }
    FrameProcess(WebProcessProxy&, BrowsingContextGroup&, const WebCore::Site&, const WebPreferences&);

    Ref<WebProcessProxy> m_process;
    WeakPtr<BrowsingContextGroup> m_browsingContextGroup;
    const WebCore::Site m_site;
};

}
