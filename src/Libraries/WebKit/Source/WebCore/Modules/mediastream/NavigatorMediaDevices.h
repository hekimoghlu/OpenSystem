/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 17, 2024.
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

#if ENABLE(MEDIA_STREAM)

#include "LocalDOMWindowProperty.h"
#include "Supplementable.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class MediaDevices;
class Navigator;

class NavigatorMediaDevices : public Supplement<Navigator>, public LocalDOMWindowProperty {
    WTF_MAKE_TZONE_ALLOCATED(NavigatorMediaDevices);
public:
    explicit NavigatorMediaDevices(LocalDOMWindow*);
    virtual ~NavigatorMediaDevices();
    static NavigatorMediaDevices* from(Navigator*);

    WEBCORE_EXPORT static MediaDevices* mediaDevices(Navigator&);
    MediaDevices* mediaDevices() const;

private:
    static ASCIILiteral supplementName();

    mutable RefPtr<MediaDevices> m_mediaDevices;
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
