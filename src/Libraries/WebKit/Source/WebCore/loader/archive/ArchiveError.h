/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 1, 2021.
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

#include <wtf/text/WTFString.h>

namespace WebCore {

enum class ArchiveError : uint8_t {
    EmptyResource,
    FileSystemError,
    InvalidFilePath,
    InvalidFrame,
    NotImplemented,
    SerializationFailure,
};

inline String errorDescription(ArchiveError error)
{
    switch (error) {
    case ArchiveError::EmptyResource:
        return "Empty resource"_s;
    case ArchiveError::FileSystemError:
        return "File system error"_s;
    case ArchiveError::InvalidFilePath:
        return "Invalid file path"_s;
    case ArchiveError::InvalidFrame:
        return "Invalid frame"_s;
    case ArchiveError::NotImplemented:
        return "Not implemented"_s;
    case ArchiveError::SerializationFailure:
        return "Serialization failure"_s;
    }
}

} // namespace WebKit
