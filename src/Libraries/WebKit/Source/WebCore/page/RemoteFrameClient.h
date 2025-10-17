/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 10, 2022.
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

#include "FrameLoaderClient.h"
#include "LayerTreeAsTextOptions.h"
#include "ScrollTypes.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

class DataSegment;
class FrameLoadRequest;
class IntSize;
class SecurityOriginData;

enum class RenderAsTextFlag : uint16_t;

struct MessageWithMessagePorts;

class RemoteFrameClient : public FrameLoaderClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(RemoteFrameClient);
public:
    virtual void frameDetached() = 0;
    virtual void sizeDidChange(IntSize) = 0;
    virtual void postMessageToRemote(FrameIdentifier source, const String& sourceOrigin, FrameIdentifier target, std::optional<SecurityOriginData> targetOrigin, const MessageWithMessagePorts&) = 0;
    virtual void changeLocation(FrameLoadRequest&&) = 0;
    virtual String renderTreeAsText(size_t baseIndent, OptionSet<RenderAsTextFlag>) = 0;
    virtual String layerTreeAsText(size_t baseIndent, OptionSet<LayerTreeAsTextOptions>) = 0;
    virtual void closePage() = 0;
    virtual void bindRemoteAccessibilityFrames(int processIdentifier, FrameIdentifier target, Vector<uint8_t>&& dataToken, CompletionHandler<void(Vector<uint8_t>, int)>&&) = 0;
    virtual void updateRemoteFrameAccessibilityOffset(FrameIdentifier target, IntPoint) = 0;
    virtual void unbindRemoteAccessibilityFrames(int) = 0;
    virtual void focus() = 0;
    virtual void unfocus() = 0;
    virtual void documentURLForConsoleLog(CompletionHandler<void(const URL&)>&&) = 0;
    virtual void updateScrollingMode(ScrollbarMode scrollingMode) = 0;
    virtual ~RemoteFrameClient() { }
};

}
