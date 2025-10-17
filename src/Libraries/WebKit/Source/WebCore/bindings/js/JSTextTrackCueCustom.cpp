/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 31, 2023.
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

#if ENABLE(VIDEO)

#include "JSTextTrackCue.h"

#include "JSDataCue.h"
#include "JSTrackCustom.h"
#include "JSVTTCue.h"
#include "TextTrack.h"
#include "WebCoreOpaqueRootInlines.h"


namespace WebCore {
using namespace JSC;

bool JSTextTrackCueOwner::isReachableFromOpaqueRoots(JSC::Handle<JSC::Unknown> handle, void*, AbstractSlotVisitor& visitor, ASCIILiteral* reason)
{
    JSTextTrackCue* jsTextTrackCue = jsCast<JSTextTrackCue*>(handle.slot()->asCell());
    TextTrackCue& textTrackCue = jsTextTrackCue->wrapped();

    if (!textTrackCue.isContextStopped() && textTrackCue.hasPendingActivity()) {
        if (UNLIKELY(reason))
            *reason = "TextTrackCue with pending activity"_s;
        return true;
    }

    // If the cue is not associated with a track, it is not reachable.
    if (!textTrackCue.track())
        return false;

    if (UNLIKELY(reason))
        *reason = "TextTrack is an opaque root"_s;

    return containsWebCoreOpaqueRoot(visitor, textTrackCue.track());
}

JSValue toJSNewlyCreated(JSGlobalObject*, JSDOMGlobalObject* globalObject, Ref<TextTrackCue>&& cue)
{
    switch (cue->cueType()) {
    case TextTrackCue::Data:
        return createWrapper<DataCue>(globalObject, WTFMove(cue));
    case TextTrackCue::WebVTT:
    case TextTrackCue::ConvertedToWebVTT:
        return createWrapper<VTTCue>(globalObject, WTFMove(cue));
    case TextTrackCue::Generic:
        return createWrapper<TextTrackCue>(globalObject, WTFMove(cue));
    }

    ASSERT_NOT_REACHED();
    return jsNull();
}

JSValue toJS(JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, TextTrackCue& cue)
{
    return wrap(lexicalGlobalObject, globalObject, cue);
}

template<typename Visitor>
void JSTextTrackCue::visitAdditionalChildren(Visitor& visitor)
{
    if (auto* textTrack = wrapped().track())
        addWebCoreOpaqueRoot(visitor, *textTrack);
}

DEFINE_VISIT_ADDITIONAL_CHILDREN(JSTextTrackCue);

} // namespace WebCore

#endif
