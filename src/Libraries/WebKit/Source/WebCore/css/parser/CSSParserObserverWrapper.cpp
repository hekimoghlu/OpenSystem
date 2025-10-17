/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 22, 2024.
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

// Copyright 2015 The Chromium Authors. All rights reserved.
// Copyright (C) 2016 Apple Inc. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//    * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//    * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "config.h"
#include "CSSParserObserverWrapper.h"

#include "CSSParserTokenRange.h"

namespace WebCore {

unsigned CSSParserObserverWrapper::startOffset(const CSSParserTokenRange& range)
{
    return m_tokenOffsets[range.begin() - m_firstParserToken];
}

unsigned CSSParserObserverWrapper::previousTokenStartOffset(const CSSParserTokenRange& range)
{
    if (range.begin() == m_firstParserToken)
        return 0;
    return m_tokenOffsets[range.begin() - m_firstParserToken - 1];
}

unsigned CSSParserObserverWrapper::endOffset(const CSSParserTokenRange& range)
{
    return m_tokenOffsets[range.end() - m_firstParserToken];
}

void CSSParserObserverWrapper::skipCommentsBefore(const CSSParserTokenRange& range, bool leaveDirectlyBefore)
{
    unsigned startIndex = range.begin() - m_firstParserToken;
    if (!leaveDirectlyBefore)
        ++startIndex;
    while (m_commentIndex < m_commentOffsets.size() && m_commentOffsets[m_commentIndex].tokensBefore < startIndex)
        ++m_commentIndex;
}

void CSSParserObserverWrapper::yieldCommentsBefore(const CSSParserTokenRange& range)
{
    unsigned startIndex = range.begin() - m_firstParserToken;
    for (; m_commentIndex < m_commentOffsets.size(); ++m_commentIndex) {
        auto& commentOffset = m_commentOffsets[m_commentIndex];
        if (commentOffset.tokensBefore > startIndex)
            break;
        m_observer.observeComment(commentOffset.startOffset, commentOffset.endOffset);
    }
}

} // namespace WebCore
