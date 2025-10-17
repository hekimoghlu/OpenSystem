/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 21, 2022.
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

#include <limits.h>
#include <wtf/Forward.h>
#include <wtf/OptionSet.h>

namespace WebCore {

// See WebDragDestinationAction and WKDragDestinationAction.
enum class DragDestinationAction : uint8_t {
    DHTML = 1,
    Edit  = 2,
    Load  = 4
};

constexpr OptionSet<DragDestinationAction> anyDragDestinationAction()
{
    return OptionSet<DragDestinationAction> { DragDestinationAction::DHTML, DragDestinationAction::Edit, DragDestinationAction::Load };
}

// See WebDragSourceAction.
enum class DragSourceAction : uint8_t {
    DHTML      = 1 << 0,
    Image      = 1 << 1,
    Link       = 1 << 2,
    Selection  = 1 << 3,
#if ENABLE(ATTACHMENT_ELEMENT)
    Attachment = 1 << 4,
#endif
    Color      = 1 << 5,
#if ENABLE(MODEL_ELEMENT)
    Model      = 1 << 6,
#endif
};

constexpr OptionSet<DragSourceAction> anyDragSourceAction()
{
    return OptionSet<DragSourceAction> {
        DragSourceAction::DHTML,
        DragSourceAction::Image,
        DragSourceAction::Link,
        DragSourceAction::Selection
#if ENABLE(ATTACHMENT_ELEMENT)
        , DragSourceAction::Attachment
#endif
        , DragSourceAction::Color
#if ENABLE(MODEL_ELEMENT)
        , DragSourceAction::Model
#endif
    };
}

// See NSDragOperation, _UIDragOperation and UIDropOperation.
enum class DragOperation : uint8_t {
    Copy    = 1,
    Link    = 2,
    Generic = 4,
    Private = 8,
    Move    = 16,
    Delete  = 32,
};

constexpr OptionSet<DragOperation> anyDragOperation()
{
    return { DragOperation::Copy, DragOperation::Link, DragOperation::Generic, DragOperation::Private, DragOperation::Move, DragOperation::Delete };
}

enum class MayExtendDragSession : bool { No, Yes };
enum class HasNonDefaultPasteboardData : bool { No, Yes };
enum class DragHandlingMethod : uint8_t { None, EditPlainText, EditRichText, UploadFile, PageLoad, SetColor, NonDefault };

} // namespace WebCore
