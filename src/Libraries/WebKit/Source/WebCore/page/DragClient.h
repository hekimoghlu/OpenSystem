/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 22, 2024.
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

#include "DragActions.h"
#include "DragData.h"
#include "DragItem.h"
#include "FloatPoint.h"
#include "IntPoint.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
    
class DataTransfer;
class Element;
class Frame;
class Image;
class LocalFrame;

#if ENABLE(ATTACHMENT_ELEMENT)
struct PromisedAttachmentInfo;
#endif

class DragClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(DragClient);
public:
    virtual bool useLegacyDragClient() { return true; }

    virtual void willPerformDragDestinationAction(DragDestinationAction, const DragData&) = 0;
    virtual void willPerformDragSourceAction(DragSourceAction, const IntPoint&, DataTransfer&) = 0;
    virtual void didConcludeEditDrag() { }
    virtual OptionSet<DragSourceAction> dragSourceActionMaskForPoint(const IntPoint& rootViewPoint) = 0;
    
    virtual void startDrag(DragItem, DataTransfer&, Frame&) = 0;
    virtual void dragEnded() { }

    virtual void beginDrag(DragItem, LocalFrame&, const IntPoint&, const IntPoint&, DataTransfer&, DragSourceAction) { }

#if PLATFORM(COCOA)
    // Mac-specific helper function to allow access to web archives and NSPasteboard extras in WebKit.
    // This is not abstract as that would require another #if PLATFORM(COCOA) for the SVGImage client empty implentation.
    virtual void declareAndWriteDragImage(const String&, Element&, const URL&, const String&, LocalFrame*) { }
#endif

    virtual ~DragClient() = default;
};
    
} // namespace WebCore
