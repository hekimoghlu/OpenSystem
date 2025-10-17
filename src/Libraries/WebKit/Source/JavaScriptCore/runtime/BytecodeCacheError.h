/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 29, 2021.
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

#include "ParserError.h"
#include <variant>

namespace JSC {

class BytecodeCacheError {
public:
    class StandardError {
    public:
        StandardError(int error)
            : m_errno(error)
        {
        }

        bool isValid() const;
        String message() const;

    private:
        int m_errno;
    };

    class WriteError {
    public:
        WriteError(size_t written, size_t expected)
            : m_written(written)
            , m_expected(expected)
        {
        }

        bool isValid() const;
        String message() const;

    private:
        size_t m_written;
        size_t m_expected;
    };

    JS_EXPORT_PRIVATE BytecodeCacheError& operator=(const ParserError&);
    JS_EXPORT_PRIVATE BytecodeCacheError& operator=(const StandardError&);
    JS_EXPORT_PRIVATE BytecodeCacheError& operator=(const WriteError&);

    JS_EXPORT_PRIVATE bool isValid() const;
    JS_EXPORT_PRIVATE String message() const;

private:
    std::variant<ParserError, StandardError, WriteError> m_error;
};

} // namespace JSC
