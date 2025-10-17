/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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
#import <WebCore/DragActions.h>
#import <WebKit/WKDragDestinationAction.h>

namespace WebKit {

inline OptionSet<WebCore::DragDestinationAction> coreDragDestinationActionMask(WKDragDestinationAction action)
{
    OptionSet<WebCore::DragDestinationAction> result;
    if (action & WKDragDestinationActionDHTML)
        result.add(WebCore::DragDestinationAction::DHTML);
    if (action & WKDragDestinationActionEdit)
        result.add(WebCore::DragDestinationAction::Edit);
    if (action & WKDragDestinationActionLoad)
        result.add(WebCore::DragDestinationAction::Load);
    return result;
}

#if USE(APPKIT)
inline OptionSet<WebCore::DragOperation> coreDragOperationMask(NSDragOperation operation)
{
    OptionSet<WebCore::DragOperation> result;
    if (operation & NSDragOperationCopy)
        result.add(WebCore::DragOperation::Copy);
    if (operation & NSDragOperationLink)
        result.add(WebCore::DragOperation::Link);
    if (operation & NSDragOperationGeneric)
        result.add(WebCore::DragOperation::Generic);
    if (operation & NSDragOperationPrivate)
        result.add(WebCore::DragOperation::Private);
    if (operation & NSDragOperationMove)
        result.add(WebCore::DragOperation::Move);
    if (operation & NSDragOperationDelete)
        result.add(WebCore::DragOperation::Delete);
    return result;
}
#endif // USE(APPKIT)

} // namespace WebKit
