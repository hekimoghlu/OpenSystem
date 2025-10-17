/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 10, 2024.
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

#if ENABLE(MHTML)

#include "Archive.h"

namespace WebCore {

class MHTMLParser;
class Page;
class FragmentedSharedBuffer;

class MHTMLArchive final : public Archive {
public:
    static Ref<MHTMLArchive> create();
    static RefPtr<MHTMLArchive> create(const URL&, FragmentedSharedBuffer&);

    static Ref<FragmentedSharedBuffer> generateMHTMLData(Page*);

    virtual ~MHTMLArchive();

private:
    friend class MHTMLParser;

    MHTMLArchive();

    bool shouldLoadFromArchiveOnly() const final { return true; }
    bool shouldOverrideBaseURL() const final { return true; }
    bool shouldUseMainResourceEncoding() const final { return false; }
    bool shouldUseMainResourceURL() const final { return false; }
};

} // namespace WebCore

#endif // ENABLE(MHTML)
