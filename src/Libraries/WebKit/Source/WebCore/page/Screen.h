/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 3, 2022.
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

#include "LocalDOMWindowProperty.h"
#include "ScriptWrappable.h"
#include <wtf/RefCounted.h>
#include <wtf/Ref.h>

namespace WebCore {

class ScreenOrientation;

class Screen final : public ScriptWrappable, public RefCounted<Screen>, public LocalDOMWindowProperty {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(Screen);
public:
    static Ref<Screen> create(LocalDOMWindow& window) { return adoptRef(*new Screen(window)); }
    ~Screen();

    int height() const;
    int width() const;
    unsigned colorDepth() const;
    int availLeft() const;
    int availTop() const;
    int availHeight() const;
    int availWidth() const;

    ScreenOrientation& orientation();

private:
    explicit Screen(LocalDOMWindow&);

    RefPtr<ScreenOrientation> m_screenOrientation;
};

} // namespace WebCore
