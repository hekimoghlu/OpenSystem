/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 30, 2025.
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

#include "APIObject.h"
#include <wtf/CompletionHandler.h>

namespace WebKit {

class WebFormSubmissionListenerProxy : public API::ObjectImpl<API::Object::Type::FormSubmissionListener> {
public:
    static Ref<WebFormSubmissionListenerProxy> create(CompletionHandler<void()>&& completionHandler)
    {
        return adoptRef(*new WebFormSubmissionListenerProxy(WTFMove(completionHandler)));
    }

    void continueSubmission();

private:
    WebFormSubmissionListenerProxy(CompletionHandler<void()>&& completionHandler)
        : m_completionHandler(WTFMove(completionHandler))
    { }
    CompletionHandler<void()> m_completionHandler;
};

} // namespace WebKit
