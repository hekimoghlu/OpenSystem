/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 20, 2024.
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

#include "Cookie.h"
#include "CookieSameSite.h"
#include "DOMHighResTimeStamp.h"
#include <optional>
#include <wtf/text/WTFString.h>

namespace WebCore {

struct CookieListItem {
    CookieListItem() = default;

    CookieListItem(Cookie&& cookie)
        : name(WTFMove(cookie.name))
        , value(WTFMove(cookie.value))
        , domain(WTFMove(cookie.domain))
        , path(WTFMove(cookie.path))
        , expires(cookie.expires)
    {
        switch (cookie.sameSite) {
        case Cookie::SameSitePolicy::Strict:
            sameSite = CookieSameSite::Strict;
            break;
        case Cookie::SameSitePolicy::Lax:
            sameSite = CookieSameSite::Lax;
            break;
        case Cookie::SameSitePolicy::None:
            sameSite = CookieSameSite::None;
            break;
        }

        // Due to how CFNetwork handles host-only cookies, we may need to prepend a '.' to the domain when
        // setting a cookie (see CookieStore::set). So we must strip this '.' when returning the cookie.
        if (domain.startsWith('.'))
            domain = domain.substring(1, domain.length() - 1);
    }

    String name;
    String value;
    String domain;
    String path;
    std::optional<DOMHighResTimeStamp> expires;
    bool secure { true };
    CookieSameSite sameSite { CookieSameSite::Strict };
};

}
