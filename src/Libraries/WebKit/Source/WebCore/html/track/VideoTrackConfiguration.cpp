/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 3, 2021.
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
#include "VideoTrackConfiguration.h"

#if ENABLE(VIDEO)

#include <wtf/JSONValues.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(VideoTrackConfiguration);

Ref<JSON::Object> VideoTrackConfiguration::toJSON() const
{
    Ref json = JSON::Object::create();
    json->setString("codec"_s, codec());
    json->setInteger("width"_s, width());
    json->setInteger("height"_s, height());
    json->setObject("colorSpace"_s, colorSpace()->toJSON());
    json->setDouble("framerate"_s, framerate());
    json->setInteger("bitrate"_s, bitrate());
    json->setBoolean("isSpatial"_s, !!spatialVideoMetadata());
    return json;
}

}

#endif
