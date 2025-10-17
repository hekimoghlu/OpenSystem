/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 3, 2023.
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

#if ENABLE(PICTURE_IN_PICTURE_API)

#include "ActiveDOMObject.h"
#include "EventTarget.h"
#include "IntSize.h"
#include <wtf/RefCounted.h>

namespace WebCore {

class PictureInPictureWindow final
    : public ActiveDOMObject
    , public EventTarget
    , public RefCounted<PictureInPictureWindow> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(PictureInPictureWindow);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<PictureInPictureWindow> create(Document&);
    virtual ~PictureInPictureWindow();

    int width() const { return m_size.width(); }
    int height() const { return m_size.height(); }
    void setSize(const IntSize&);
    void close();

private:
    PictureInPictureWindow(Document&);

    // EventTarget.
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }
    enum EventTargetInterfaceType eventTargetInterface() const override { return EventTargetInterfaceType::PictureInPictureWindow; };
    ScriptExecutionContext* scriptExecutionContext() const override { return ActiveDOMObject::scriptExecutionContext(); };

    IntSize m_size;
};

} // namespace WebCore

#endif // ENABLE(PICTURE_IN_PICTURE_API)
