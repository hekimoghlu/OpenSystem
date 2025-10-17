/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 5, 2024.
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
#include "VideoColorSpace.h"

#if ENABLE(VIDEO)

#include "JSVideoColorPrimaries.h"
#include "JSVideoMatrixCoefficients.h"
#include "JSVideoTransferCharacteristics.h"
#include <wtf/JSONValues.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(VideoColorSpace);

Ref<JSON::Object> VideoColorSpace::toJSON() const
{
    Ref json = JSON::Object::create();

    if (auto& primaries = this->primaries())
        json->setString("primaries"_s, convertEnumerationToString(*primaries));
    if (auto& transfer = this->transfer())
        json->setString("transfer"_s, convertEnumerationToString(*transfer));
    if (auto& matrix = this->matrix())
        json->setString("matrix"_s, convertEnumerationToString(*matrix));
    if (auto& fullRange = this->fullRange())
        json->setBoolean("fullRange"_s, *fullRange);

    return json;
}

}

#endif
