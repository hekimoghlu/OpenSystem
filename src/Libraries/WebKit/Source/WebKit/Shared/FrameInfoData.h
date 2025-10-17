/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 20, 2022.
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

#include "WebFrameMetrics.h"
#include <WebCore/FrameIdentifier.h>
#include <WebCore/ResourceRequest.h>
#include <WebCore/ScriptExecutionContextIdentifier.h>
#include <WebCore/SecurityOriginData.h>
#include <wtf/ProcessID.h>

namespace WebKit {

enum class FrameType : bool { Local, Remote };

struct FrameInfoData {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    bool isMainFrame { false };
    FrameType frameType { FrameType::Local };
    WebCore::ResourceRequest request;
    WebCore::SecurityOriginData securityOrigin;
    String frameName;
    Markable<WebCore::FrameIdentifier> frameID;
    Markable<WebCore::FrameIdentifier> parentFrameID;
    Markable<WebCore::ScriptExecutionContextIdentifier> documentID;
    ProcessID processID;
    bool isFocused { false };
    bool errorOccurred { false };
    WebFrameMetrics frameMetrics { };
};

}
