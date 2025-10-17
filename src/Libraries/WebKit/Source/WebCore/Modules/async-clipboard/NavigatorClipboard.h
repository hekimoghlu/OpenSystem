/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 14, 2025.
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
#include <wtf/CheckedRef.h>
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class Clipboard;
class Navigator;

class NavigatorClipboard final : public Supplement<Navigator> {
    WTF_MAKE_TZONE_ALLOCATED(NavigatorClipboard);
public:
    explicit NavigatorClipboard(Navigator&);
    ~NavigatorClipboard();

    static RefPtr<Clipboard> clipboard(Navigator&);
    RefPtr<Clipboard> clipboard();

private:
    static NavigatorClipboard* from(Navigator&);
    static ASCIILiteral supplementName();

    RefPtr<Clipboard> m_clipboard;
    CheckedRef<Navigator> m_navigator;
};

}
