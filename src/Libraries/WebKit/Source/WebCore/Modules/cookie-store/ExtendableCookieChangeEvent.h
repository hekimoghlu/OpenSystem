/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 17, 2024.
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

#include "ExtendableCookieChangeEventInit.h"
#include "ExtendableEvent.h"
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

struct CookieListItem;

class ExtendableCookieChangeEvent final : public ExtendableEvent {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ExtendableCookieChangeEvent);
public:
    static Ref<ExtendableCookieChangeEvent> create(const AtomString& type, ExtendableCookieChangeEventInit&&, IsTrusted = IsTrusted::No);
    ~ExtendableCookieChangeEvent();

    const Vector<CookieListItem>& changed() const { return m_changed; }
    const Vector<CookieListItem>& deleted() const { return m_deleted; }

private:
    ExtendableCookieChangeEvent(const AtomString& type, ExtendableCookieChangeEventInit&&, IsTrusted);

    Vector<CookieListItem> m_changed;
    Vector<CookieListItem> m_deleted;
};

} // namespace WebCore
