/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 10, 2024.
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
#import "config.h"
#import "DataTransfer.h"

#if PLATFORM(MAC)

#import "CachedImage.h"
#import "Document.h"
#import "DragImage.h"
#import "Element.h"
#import "FrameDestructionObserverInlines.h"

namespace WebCore {

// FIXME: Need to refactor and figure out how to handle the flipping in a more sensible way so we can
// use the default DataTransfer::dragImage from DataTransfer.cpp. Note also that this handles cases that
// DataTransfer::dragImage in DataTransfer.cpp does not handle correctly, so must resolve that as well.
DragImageRef DataTransfer::createDragImage(IntPoint& location) const
{
    DragImageRef result = nil;
    if (m_dragImageElement) {
        if (RefPtr frame = m_dragImageElement->document().frame()) {
            IntRect imageRect;
            IntRect elementRect;
            result = createDragImageForImage(*frame, dragImageElement().releaseNonNull(), imageRect, elementRect);
            // Client specifies point relative to element, not the whole image, which may include child
            // layers spread out all over the place.
            location.setX(elementRect.x() - imageRect.x() + m_dragLocation.x());
            location.setY(imageRect.height() - (elementRect.y() - imageRect.y() + m_dragLocation.y()));
        }
    } else if (m_dragImage) {
        result = m_dragImage->protectedImage()->adapter().snapshotNSImage();
        
        location = m_dragLocation;
        location.setY([result size].height - location.y());
    }
    return result;
}

}

#endif // PLATFORM(MAC)
