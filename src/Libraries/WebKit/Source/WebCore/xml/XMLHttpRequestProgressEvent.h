/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 15, 2023.
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

#include "ProgressEvent.h"

namespace WebCore {

class XMLHttpRequestProgressEvent final : public ProgressEvent {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(XMLHttpRequestProgressEvent);
public:
    static Ref<XMLHttpRequestProgressEvent> create(const AtomString& type, bool lengthComputable = false, unsigned long long loaded = 0, unsigned long long total = 0)
    {
        return adoptRef(*new XMLHttpRequestProgressEvent(type, lengthComputable, loaded, total));
    }
    // Those 2 synonyms are included for compatibility with Firefox.
    unsigned long long position() const { return loaded(); }
    unsigned long long totalSize() const { return total(); }

private:
    XMLHttpRequestProgressEvent(const AtomString& type, bool lengthComputable, unsigned long long loaded, unsigned long long total)
        : ProgressEvent(EventInterfaceType::XMLHttpRequestProgressEvent, type, lengthComputable, loaded, total)
    {
    }
};

} // namespace WebCore
