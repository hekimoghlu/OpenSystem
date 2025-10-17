/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 24, 2023.
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

#include "InputType.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

// Base of input types that dispatches a simulated click on space/return key.
class BaseClickableWithKeyInputType : public InputType {
    WTF_MAKE_TZONE_ALLOCATED(BaseClickableWithKeyInputType);
public:
    static ShouldCallBaseEventHandler handleKeydownEvent(HTMLInputElement&, KeyboardEvent&);
    static void handleKeypressEvent(HTMLInputElement&, KeyboardEvent&);
    static void handleKeyupEvent(InputType&, KeyboardEvent&);
    static bool accessKeyAction(HTMLInputElement&, bool sendMouseEvents);
    
protected:
    explicit BaseClickableWithKeyInputType(Type type, HTMLInputElement& element)
        : InputType(type, element)
    {
    }

private:
    ShouldCallBaseEventHandler handleKeydownEvent(KeyboardEvent&) final;
    void handleKeypressEvent(KeyboardEvent&) final;
    void handleKeyupEvent(KeyboardEvent&) final;
    bool accessKeyAction(bool sendMouseEvents) final;
};

} // namespace WebCore
