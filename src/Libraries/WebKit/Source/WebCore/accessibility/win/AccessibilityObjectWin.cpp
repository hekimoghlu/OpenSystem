/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 19, 2024.
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
#include "config.h"
#include "AccessibilityObject.h"

#include "AXObjectCache.h"

namespace WebCore {

void AccessibilityObject::detachPlatformWrapper(AccessibilityDetachmentType)
{
    // On Windows, AccessibilityObjects are created when get_accChildCount is
    // called, but they are not wrapped until get_accChild is called, so this
    // object may not have a wrapper.
    if (AccessibilityObjectWrapper* wrapper = this->wrapper())
        wrapper->detach();
}

bool AccessibilityObject::accessibilityIgnoreAttachment() const
{
    return false;
}

AccessibilityObjectInclusion AccessibilityObject::accessibilityPlatformIncludesObject() const
{
    if (isMenuListPopup() || isMenuListOption())
        return AccessibilityObjectInclusion::IncludeObject;

    return AccessibilityObjectInclusion::DefaultBehavior;
}

} // namespace WebCore
