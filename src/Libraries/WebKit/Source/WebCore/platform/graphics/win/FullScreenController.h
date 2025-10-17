/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 9, 2023.
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

#include <memory>

namespace WebCore {

class FullScreenControllerClient;

class FullScreenController {
public:
    WEBCORE_EXPORT FullScreenController(FullScreenControllerClient*);
    WEBCORE_EXPORT ~FullScreenController();

public:
    WEBCORE_EXPORT void enterFullScreen();
    WEBCORE_EXPORT void exitFullScreen();
    WEBCORE_EXPORT void repaintCompleted();
    
    WEBCORE_EXPORT bool isFullScreen() const;

    WEBCORE_EXPORT void close();

protected:
    void enterFullScreenRepaintCompleted();
    void exitFullScreenRepaintCompleted();

    class Private;
    friend class Private;
    std::unique_ptr<FullScreenController::Private> m_private;
};

} // namespace WebCore

#endif // ENABLE(FULLSCREEN_API)
