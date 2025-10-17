/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 29, 2021.
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
#ifndef PopupMenu_h
#define PopupMenu_h

#include <wtf/RefCounted.h>

namespace WebCore {

class IntRect;
class LocalFrameView;

class PopupMenu : public RefCounted<PopupMenu> {
public:
    virtual ~PopupMenu() = default;
    virtual void show(const IntRect&, LocalFrameView&, int selectedIndex) = 0;
    virtual void hide() = 0;
    virtual void updateFromElement() = 0;
    virtual void disconnectClient() = 0;
};

}

#endif // PopupMenu_h
