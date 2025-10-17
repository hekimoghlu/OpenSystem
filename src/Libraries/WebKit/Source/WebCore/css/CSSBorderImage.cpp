/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 12, 2022.
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
#include "CSSBorderImage.h"

#include "CSSValueList.h"

namespace WebCore {

Ref<CSSValueList> createBorderImageValue(RefPtr<CSSValue>&& image, RefPtr<CSSValue>&& imageSlice, RefPtr<CSSValue>&& borderSlice, RefPtr<CSSValue>&& outset, RefPtr<CSSValue>&& repeat)
{
    CSSValueListBuilder list;
    if (image)
        list.append(*image);
    if (borderSlice || outset) {
        CSSValueListBuilder listSlash;
        if (imageSlice)
            listSlash.append(imageSlice.releaseNonNull());
        if (borderSlice)
            listSlash.append(borderSlice.releaseNonNull());
        if (outset)
            listSlash.append(outset.releaseNonNull());
        list.append(CSSValueList::createSlashSeparated(WTFMove(listSlash)));
    } else if (imageSlice)
        list.append(imageSlice.releaseNonNull());
    if (repeat)
        list.append(repeat.releaseNonNull());
    return CSSValueList::createSpaceSeparated(WTFMove(list));
}

} // namespace WebCore
