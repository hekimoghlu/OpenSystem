/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 10, 2024.
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
#include "ModelPlayer.h"

#include "Color.h"
#include "TransformationMatrix.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ModelPlayer);

ModelPlayer::~ModelPlayer() = default;

void ModelPlayer::setBackgroundColor(Color)
{
}

void ModelPlayer::setEntityTransform(TransformationMatrix)
{
}

bool ModelPlayer::supportsMouseInteraction()
{
    return false;
}

bool ModelPlayer::supportsDragging()
{
    return true;
}

bool ModelPlayer::supportsTransform(TransformationMatrix)
{
    return false;
}

void ModelPlayer::setInteractionEnabled(bool)
{
}

String ModelPlayer::inlinePreviewUUIDForTesting() const
{
    return emptyString();
}

#if ENABLE(MODEL_PROCESS)
void ModelPlayer::setAutoplay(bool)
{
}

void ModelPlayer::setLoop(bool)
{
}

void ModelPlayer::setPlaybackRate(double, CompletionHandler<void(double effectivePlaybackRate)>&& completionHandler)
{
    completionHandler(1.0);
}

double ModelPlayer::duration() const
{
    return 0;
}

bool ModelPlayer::paused() const
{
    return true;
}

void ModelPlayer::setPaused(bool, CompletionHandler<void(bool succeeded)>&& completionHandler)
{
    completionHandler(false);
}

Seconds ModelPlayer::currentTime() const
{
    return 0_s;
}

void ModelPlayer::setCurrentTime(Seconds, CompletionHandler<void()>&& completionHandler)
{
    completionHandler();
}

void ModelPlayer::setEnvironmentMap(Ref<SharedBuffer>&&)
{
}

void ModelPlayer::setHasPortal(bool)
{
}
#endif // ENABLE(MODEL_PROCESS)

}
