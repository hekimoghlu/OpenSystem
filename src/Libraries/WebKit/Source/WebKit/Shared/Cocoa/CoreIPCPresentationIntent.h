/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 4, 2022.
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

#if PLATFORM(COCOA)

#include <wtf/ArgumentCoder.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

OBJC_CLASS NSPresentationIntent;

namespace WebKit {

class CoreIPCPresentationIntent {
    WTF_MAKE_TZONE_ALLOCATED(CoreIPCPresentationIntent);
public:
    CoreIPCPresentationIntent(NSPresentationIntent *);

    RetainPtr<id> toID() const;

private:
    friend struct IPC::ArgumentCoder<CoreIPCPresentationIntent, void>;

    CoreIPCPresentationIntent(int64_t intentKind, int64_t identity, std::unique_ptr<CoreIPCPresentationIntent>&& parentIntent, int64_t column, Vector<int64_t>&& columnAlignments, int64_t columnCount, int64_t headerLevel, const String& languageHint, int64_t ordinal, int64_t row)
        : m_intentKind(intentKind)
        , m_identity(identity)
        , m_parentIntent(WTFMove(parentIntent))
        , m_column(column)
        , m_columnAlignments(WTFMove(columnAlignments))
        , m_columnCount(columnCount)
        , m_headerLevel(headerLevel)
        , m_languageHint(languageHint)
        , m_ordinal(ordinal)
        , m_row(row)
    {
    }

    int64_t m_intentKind { 0 };
    int64_t m_identity { 0 };
    std::unique_ptr<CoreIPCPresentationIntent> m_parentIntent;

    int64_t m_column { 0 };
    Vector<int64_t> m_columnAlignments;
    int64_t m_columnCount { 0 };
    int64_t m_headerLevel { 0 };
    String m_languageHint;
    int64_t m_ordinal { 0 };
    int64_t m_row { 0 };
};

} // namespace WebKit

#endif // PLATFORM(COCOA)
