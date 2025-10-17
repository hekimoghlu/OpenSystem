/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 19, 2021.
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

#include "HTMLDocument.h"

namespace WebCore {

class TextDocument final : public HTMLDocument {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(TextDocument);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(TextDocument);
public:
    static Ref<TextDocument> create(LocalFrame* frame, const Settings& settings, const URL& url, std::optional<ScriptExecutionContextIdentifier> identifier)
    {
        auto document = adoptRef(*new TextDocument(frame, settings, url, identifier));
        document->addToContextsMap();
        return document;
    }

private:
    TextDocument(LocalFrame*, const Settings&, const URL&, std::optional<ScriptExecutionContextIdentifier>);
    
    Ref<DocumentParser> createParser() override;
};

} // namespace WebCore
