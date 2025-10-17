/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 9, 2022.
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
#include "PointerEventTypeNames.h"

#include <wtf/NeverDestroyed.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

const String& mousePointerEventType()
{
    static NeverDestroyed<const String> mouseType(MAKE_STATIC_STRING_IMPL("mouse"));
    return mouseType;
}

const String& penPointerEventType()
{
    static NeverDestroyed<const String> penType(MAKE_STATIC_STRING_IMPL("pen"));
    return penType;
}

const String& touchPointerEventType()
{
    static NeverDestroyed<const String> touchType(MAKE_STATIC_STRING_IMPL("touch"));
    return touchType;
}

} // namespace WebCore
