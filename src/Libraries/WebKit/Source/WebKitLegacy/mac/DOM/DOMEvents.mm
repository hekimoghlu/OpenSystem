/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 16, 2025.
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
#import "DOMEventInternal.h"

#import "DOMKeyboardEvent.h"
#import "DOMMouseEvent.h"
#import "DOMMutationEvent.h"
#import "DOMOverflowEvent.h"
#import "DOMProgressEvent.h"
#import "DOMTextEvent.h"
#import "DOMWheelEvent.h"
#import <WebCore/Event.h>
#import <WebCore/EventNames.h>

using WebCore::eventNames;

Class kitClass(WebCore::Event* impl)
{
    switch (impl->interfaceType()) {
    case WebCore::EventInterfaceType::KeyboardEvent:
        return [DOMKeyboardEvent class];
    case WebCore::EventInterfaceType::MouseEvent:
        return [DOMMouseEvent class];
    case WebCore::EventInterfaceType::MutationEvent:
        return [DOMMutationEvent class];
    case WebCore::EventInterfaceType::OverflowEvent:
        return [DOMOverflowEvent class];
    case WebCore::EventInterfaceType::ProgressEvent:
    case WebCore::EventInterfaceType::XMLHttpRequestProgressEvent:
        return [DOMProgressEvent class];
    case WebCore::EventInterfaceType::TextEvent:
        return [DOMTextEvent class];
    case WebCore::EventInterfaceType::WheelEvent:
        return [DOMWheelEvent class];
    default:
        if (impl->isUIEvent())
            return [DOMUIEvent class];

        return [DOMEvent class];
    }
}
