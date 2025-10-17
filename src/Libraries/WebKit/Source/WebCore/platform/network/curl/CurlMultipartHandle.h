/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 13, 2022.
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

#include <wtf/CheckedRef.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class CurlMultipartHandleClient;
class CurlResponse;
class SharedBuffer;

class CurlMultipartHandle {
    WTF_MAKE_TZONE_ALLOCATED(CurlMultipartHandle);
public:
    WEBCORE_EXPORT static std::unique_ptr<CurlMultipartHandle> createIfNeeded(CurlMultipartHandleClient&, const CurlResponse&);

    CurlMultipartHandle(CurlMultipartHandleClient&, CString&&);
    ~CurlMultipartHandle() { }

    WEBCORE_EXPORT void didReceiveMessage(std::span<const uint8_t>);
    WEBCORE_EXPORT void didCompleteMessage();

    WEBCORE_EXPORT void completeHeaderProcessing();

    bool completed() { return m_didCompleteMessage; }
    bool hasError() const { return m_hasError; }

private:
    enum class State {
        FindBoundaryStart,
        InHeader,
        WaitingForHeaderProcessing,
        InBody,
        WaitingForTerminate,
        Terminating,
        End
    };

    enum class ParseHeadersResult {
        Success,
        NeedMoreData,
        HeaderSizeTooLarge
    };

    struct FindBoundaryResult {
        bool isSyntaxError { false };
        bool hasBoundary { false };
        bool hasCloseDelimiter { false };
        size_t processed { 0 };
        size_t dataEnd { 0 };
    };

    bool processContent();
    FindBoundaryResult findBoundary();
    ParseHeadersResult parseHeadersIfPossible();

    CheckedRef<CurlMultipartHandleClient> m_client;

    CString m_boundary;
    Vector<uint8_t> m_buffer;
    Vector<String> m_headers;

    State m_state { State::FindBoundaryStart };
    bool m_didCompleteMessage { false };
    bool m_hasError { false };
};

} // namespace WebCore
