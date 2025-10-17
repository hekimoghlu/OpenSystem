/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 28, 2024.
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

#if ENABLE(ARKIT_INLINE_PREVIEW)

#include "ModelIdentifier.h"
#include <WebCore/ElementContext.h>
#include <WebCore/GraphicsLayer.h>
#include <WebCore/HTMLModelElementCamera.h>
#include <WebCore/ResourceError.h>
#include <wtf/MachSendRight.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/URL.h>

OBJC_CLASS ASVInlinePreview;

#if ENABLE(ARKIT_INLINE_PREVIEW_IOS)
OBJC_CLASS WKModelView;
#endif

namespace WebKit {

class WebPageProxy;

class ModelElementController : public RefCountedAndCanMakeWeakPtr<ModelElementController> {
    WTF_MAKE_TZONE_ALLOCATED(ModelElementController);
public:
    static Ref<ModelElementController> create(WebPageProxy&);

    WebPageProxy* page();

#if ENABLE(ARKIT_INLINE_PREVIEW)
    void getCameraForModelElement(ModelIdentifier, CompletionHandler<void(Expected<WebCore::HTMLModelElementCamera, WebCore::ResourceError>)>&&);
    void setCameraForModelElement(ModelIdentifier, WebCore::HTMLModelElementCamera, CompletionHandler<void(bool)>&&);
    void isPlayingAnimationForModelElement(ModelIdentifier, CompletionHandler<void(Expected<bool, WebCore::ResourceError>)>&&);
    void setAnimationIsPlayingForModelElement(ModelIdentifier, bool, CompletionHandler<void(bool)>&&);
    void isLoopingAnimationForModelElement(ModelIdentifier, CompletionHandler<void(Expected<bool, WebCore::ResourceError>)>&&);
    void setIsLoopingAnimationForModelElement(ModelIdentifier, bool, CompletionHandler<void(bool)>&&);
    void animationDurationForModelElement(ModelIdentifier, CompletionHandler<void(Expected<Seconds, WebCore::ResourceError>)>&&);
    void animationCurrentTimeForModelElement(ModelIdentifier, CompletionHandler<void(Expected<Seconds, WebCore::ResourceError>)>&&);
    void setAnimationCurrentTimeForModelElement(ModelIdentifier, Seconds, CompletionHandler<void(bool)>&&);
    void hasAudioForModelElement(ModelIdentifier, CompletionHandler<void(Expected<bool, WebCore::ResourceError>)>&&);
    void isMutedForModelElement(ModelIdentifier, CompletionHandler<void(Expected<bool, WebCore::ResourceError>)>&&);
    void setIsMutedForModelElement(ModelIdentifier, bool, CompletionHandler<void(bool)>&&);
#endif
#if ENABLE(ARKIT_INLINE_PREVIEW_IOS)
    void takeModelElementFullscreen(ModelIdentifier, const URL&);
    void setInteractionEnabledForModelElement(ModelIdentifier, bool);
#endif
#if ENABLE(ARKIT_INLINE_PREVIEW_MAC)
    void modelElementCreateRemotePreview(String, WebCore::FloatSize, CompletionHandler<void(Expected<std::pair<String, uint32_t>, WebCore::ResourceError>)>&&);
    void modelElementLoadRemotePreview(String, URL, CompletionHandler<void(std::optional<WebCore::ResourceError>&&)>&&);
    void modelElementDestroyRemotePreview(String);
    void modelElementSizeDidChange(const String& uuid, WebCore::FloatSize, CompletionHandler<void(Expected<MachSendRight, WebCore::ResourceError>)>&&);
    void handleMouseDownForModelElement(const String&, const WebCore::LayoutPoint&, MonotonicTime);
    void handleMouseMoveForModelElement(const String&, const WebCore::LayoutPoint&, MonotonicTime);
    void handleMouseUpForModelElement(const String&, const WebCore::LayoutPoint&, MonotonicTime);
    void inlinePreviewUUIDs(CompletionHandler<void(Vector<String>&&)>&&);
#endif

private:
    explicit ModelElementController(WebPageProxy&);

#if ENABLE(ARKIT_INLINE_PREVIEW)
    ASVInlinePreview * previewForModelIdentifier(ModelIdentifier);
#endif

#if ENABLE(ARKIT_INLINE_PREVIEW_IOS)
    WKModelView * modelViewForModelIdentifier(ModelIdentifier);
#endif

    WeakPtr<WebPageProxy> m_webPageProxy;
#if ENABLE(ARKIT_INLINE_PREVIEW_MAC)
    RetainPtr<ASVInlinePreview> previewForUUID(const String&);
    HashMap<String, RetainPtr<ASVInlinePreview>> m_inlinePreviews;
#endif
};

} // namespace WebKit

#endif
