/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 6, 2024.
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

#include "DOMStringList.h"
#include "EventTarget.h"
#include "ExceptionOr.h"
#include "ScriptWrappable.h"
#include <wtf/WeakPtr.h>

namespace WebCore {

class DOMWindow;
class Frame;
class LocalDOMWindow;

class Location final : public ScriptWrappable, public RefCounted<Location> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(Location);
public:
    static Ref<Location> create(DOMWindow& window) { return adoptRef(*new Location(window)); }

    ExceptionOr<void> setHref(LocalDOMWindow& incumbentWindow, LocalDOMWindow& firstWindow, const String&);
    String href() const;

    ExceptionOr<void> assign(LocalDOMWindow& activeWindow, LocalDOMWindow& firstWindow, const String&);
    ExceptionOr<void> replace(LocalDOMWindow& activeWindow, LocalDOMWindow& firstWindow, const String&);
    void reload(LocalDOMWindow& activeWindow);

    ExceptionOr<void> setProtocol(LocalDOMWindow& incumbentWindow, LocalDOMWindow& firstWindow, const String&);
    String protocol() const;
    ExceptionOr<void> setHost(LocalDOMWindow& incumbentWindow, LocalDOMWindow& firstWindow, const String&);
    WEBCORE_EXPORT String host() const;
    ExceptionOr<void> setHostname(LocalDOMWindow& incumbentWindow, LocalDOMWindow& firstWindow, const String&);
    String hostname() const;
    ExceptionOr<void> setPort(LocalDOMWindow& incumbentWindow, LocalDOMWindow& firstWindow, const String&);
    String port() const;
    ExceptionOr<void> setPathname(LocalDOMWindow& incumbentWindow, LocalDOMWindow& firstWindow, const String&);
    String pathname() const;
    ExceptionOr<void> setSearch(LocalDOMWindow& incumbentWindow, LocalDOMWindow& firstWindow, const String&);
    String search() const;
    ExceptionOr<void> setHash(LocalDOMWindow& incumbentWindow, LocalDOMWindow& firstWindow, const String&);
    String hash() const;
    String origin() const;

    String toString() const { return href(); }

    Ref<DOMStringList> ancestorOrigins() const;

    DOMWindow* window() { return m_window.get(); }
    RefPtr<DOMWindow> protectedWindow();

    const URL& url() const;

private:
    explicit Location(DOMWindow&);

    ExceptionOr<void> setLocation(LocalDOMWindow& incumbentWindow, LocalDOMWindow& firstWindow, const String&);

    Frame* frame();
    const Frame* frame() const;

    WeakPtr<DOMWindow, WeakPtrImplWithEventTargetData> m_window;
};

} // namespace WebCore
