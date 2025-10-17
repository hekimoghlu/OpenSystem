/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 24, 2023.
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
#include "ContentTypeUtilities.h"

#include "FourCC.h"
#include <wtf/Algorithms.h>

namespace WebCore {

bool contentTypeMeetsContainerAndCodecTypeRequirements(const ContentType& type, const std::optional<Vector<String>>& allowedMediaContainerTypes, const std::optional<Vector<String>>& allowedMediaCodecTypes)
{
    if (allowedMediaContainerTypes && !allowedMediaContainerTypes->contains(type.containerType()))
        return false;

    if (!allowedMediaCodecTypes)
        return true;

    return WTF::allOf(type.codecs(), [&] (auto& codec) {
        return WTF::anyOf(*allowedMediaCodecTypes, [&] (auto& allowedCodec) {
            return codec.startsWith(allowedCodec);
        });
    });
}

}
