/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 2, 2021.
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

#include "Supplementable.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class DeferredPromise;
class Document;
class Navigator;
enum class IsLoggedIn : uint8_t;

class NavigatorLoginStatus final : public Supplement<Navigator>, public CanMakeWeakPtr<NavigatorLoginStatus> {
    WTF_MAKE_TZONE_ALLOCATED(NavigatorLoginStatus);
public:
    explicit NavigatorLoginStatus(Navigator& navigator)
        : m_navigator(navigator)
    {
    }
    static void setStatus(Navigator&, IsLoggedIn, Ref<DeferredPromise>&&);
    static void isLoggedIn(Navigator&, Ref<DeferredPromise>&&);

private:
    void setStatus(IsLoggedIn, Ref<DeferredPromise>&&);
    void isLoggedIn(Ref<DeferredPromise>&&);

    static NavigatorLoginStatus* from(Navigator&);
    static ASCIILiteral supplementName();
    bool hasSameOrigin() const;

    Navigator& m_navigator;
};

} // namespace WebCore
