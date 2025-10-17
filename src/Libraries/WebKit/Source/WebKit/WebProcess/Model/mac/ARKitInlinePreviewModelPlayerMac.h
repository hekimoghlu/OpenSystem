/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 19, 2022.
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

#if ENABLE(ARKIT_INLINE_PREVIEW_MAC)

#include "ARKitInlinePreviewModelPlayer.h"
#include <WebCore/ModelPlayer.h>
#include <WebCore/ModelPlayerClient.h>
#include <wtf/Compiler.h>
#include <wtf/RetainPtr.h>

OBJC_CLASS ASVInlinePreview;

namespace WebKit {

class ARKitInlinePreviewModelPlayerMac final : public ARKitInlinePreviewModelPlayer {
public:
    static Ref<ARKitInlinePreviewModelPlayerMac> create(WebPage&, WebCore::ModelPlayerClient&);
    virtual ~ARKitInlinePreviewModelPlayerMac();

    static void setModelElementCacheDirectory(const String&);
    static const String& modelElementCacheDirectory();

private:
    ARKitInlinePreviewModelPlayerMac(WebPage&, WebCore::ModelPlayerClient&);

    std::optional<ModelIdentifier> modelIdentifier() override;

    // WebCore::ModelPlayer overrides.
    void load(WebCore::Model&, WebCore::LayoutSize) override;
    void sizeDidChange(WebCore::LayoutSize) override;
    PlatformLayer* layer() override;
    bool supportsMouseInteraction() override;
    bool supportsDragging() override;
    void handleMouseDown(const WebCore::LayoutPoint&, MonotonicTime) override;
    void handleMouseMove(const WebCore::LayoutPoint&, MonotonicTime) override;
    void handleMouseUp(const WebCore::LayoutPoint&, MonotonicTime) override;
    String inlinePreviewUUIDForTesting() const override;

    void createFile(WebCore::Model&);
    void clearFile();

    void createPreviewsForModelWithURL(const URL&);
    void didCreateRemotePreviewForModelWithURL(const URL&);

    WebCore::LayoutSize m_size;
    String m_filePath;
    RetainPtr<ASVInlinePreview> m_inlinePreview;
};

}

#endif
