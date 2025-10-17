/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 9, 2022.
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
#import "config.h"
#import "WebExtensionFrameIdentifier.h"

#import "WKFrameInfoPrivate.h"
#import "_WKFrameHandle.h"

namespace WebKit {

WebExtensionFrameIdentifier toWebExtensionFrameIdentifier(WKFrameInfo *frameInfo)
{
    if (frameInfo.isMainFrame)
        return WebExtensionFrameConstants::MainFrameIdentifier;

    // FIXME: <rdar://117932176> Stop using FrameIdentifier/_WKFrameHandle for WebExtensionFrameIdentifier,
    // which needs to be just one number and probably should only be generated in the UI process
    // to prevent collisions with numbers generated in different web content processes, especially with site isolation.
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    auto identifier = frameInfo._handle.frameID;
ALLOW_DEPRECATED_DECLARATIONS_END
    if (!WebExtensionFrameIdentifier::isValidIdentifier(identifier)) {
        ASSERT_NOT_REACHED();
        return WebExtensionFrameConstants::NoneIdentifier;
    }

    return WebExtensionFrameIdentifier { identifier };
}

}
