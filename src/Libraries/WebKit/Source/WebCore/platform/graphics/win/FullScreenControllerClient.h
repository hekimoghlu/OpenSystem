/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 6, 2024.
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

#if ENABLE(FULLSCREEN_API)

namespace WebCore {

class FullScreenControllerClient {
public:
    virtual HWND fullScreenClientWindow() const = 0;
    virtual HWND fullScreenClientParentWindow() const = 0;
    virtual void fullScreenClientSetParentWindow(HWND) = 0;
    virtual void fullScreenClientWillEnterFullScreen() = 0;
    virtual void fullScreenClientDidEnterFullScreen() = 0;
    virtual void fullScreenClientWillExitFullScreen() = 0;
    virtual void fullScreenClientDidExitFullScreen() = 0;
    virtual void fullScreenClientForceRepaint() = 0;
    virtual void fullScreenClientSaveScrollPosition() = 0;
    virtual void fullScreenClientRestoreScrollPosition() = 0;
protected:
    virtual ~FullScreenControllerClient() = default;
};

} // namespace WebCore

#endif // ENABLE(FULLSCREEN_API)
