/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 27, 2021.
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

#include "DOMHighResTimeStamp.h"
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class Document;

class IdleDeadline final : public RefCounted<IdleDeadline> {
public:

    enum class DidTimeout : bool { No, Yes };
    static Ref<IdleDeadline> create(DidTimeout didTimeout)
    {
        return adoptRef(*new IdleDeadline(didTimeout));
    }

    DOMHighResTimeStamp timeRemaining(Document&) const;
    bool didTimeout(Document&) const { return m_didTimeout == DidTimeout::Yes; }

private:
    IdleDeadline(DidTimeout didTimeout)
        : m_didTimeout(didTimeout)
    { }

    const DidTimeout m_didTimeout;
};

} // namespace WebCore
