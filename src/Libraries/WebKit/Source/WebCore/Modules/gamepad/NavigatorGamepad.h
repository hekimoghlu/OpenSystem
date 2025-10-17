/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 15, 2024.
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

#if ENABLE(GAMEPAD)

#include "Supplementable.h"
#include <wtf/CheckedRef.h>
#include <wtf/MonotonicTime.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {
class NavigatorGamepad;
}

namespace WebCore {

class Gamepad;
class Navigator;
class Page;
class PlatformGamepad;
template<typename> class ExceptionOr;

class NavigatorGamepad : public Supplement<Navigator> {
    WTF_MAKE_TZONE_ALLOCATED(NavigatorGamepad);
public:
    explicit NavigatorGamepad(Navigator&);
    virtual ~NavigatorGamepad();

    static NavigatorGamepad& from(Navigator&);

    Navigator& navigator() const;

    // The array of Gamepads might be sparse.
    // Null checking each entry is necessary.
    static ExceptionOr<const Vector<RefPtr<Gamepad>>&> getGamepads(Navigator&);

    void gamepadConnected(PlatformGamepad&);
    void gamepadDisconnected(PlatformGamepad&);

    Ref<Gamepad> gamepadFromPlatformGamepad(PlatformGamepad&);

    WEBCORE_EXPORT static void setGamepadsRecentlyAccessedThreshold(Seconds);
    static Seconds gamepadsRecentlyAccessedThreshold();

    RefPtr<Page> protectedPage() const;

private:
    static ASCIILiteral supplementName();
    Ref<Navigator> protectedNavigator() const;

    void gamepadsBecameVisible();
    void maybeNotifyRecentAccess();

    const Vector<RefPtr<Gamepad>>& gamepads();

    CheckedRef<Navigator> m_navigator;
    Vector<RefPtr<Gamepad>> m_gamepads;
};

} // namespace WebCore

#endif // ENABLE(GAMEPAD)
