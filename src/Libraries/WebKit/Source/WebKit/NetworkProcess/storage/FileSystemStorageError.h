/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 6, 2025.
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

#include <WebCore/ExceptionOr.h>

namespace WebKit {

enum class FileSystemStorageError : uint8_t {
    AccessHandleActive,
    BackendNotSupported,
    FileNotFound,
    InvalidDataType,
    InvalidModification,
    InvalidName,
    InvalidState,
    MissingArgument,
    TypeMismatch,
    Unknown
};

inline WebCore::Exception convertToException(FileSystemStorageError error)
{
    switch (error) {
    case FileSystemStorageError::AccessHandleActive:
        return WebCore::Exception { WebCore::ExceptionCode::InvalidStateError, "Some AccessHandle is active"_s };
    case FileSystemStorageError::BackendNotSupported:
        return WebCore::Exception { WebCore::ExceptionCode::NotSupportedError, "Backend does not support this operation"_s };
    case FileSystemStorageError::FileNotFound:
        return WebCore::Exception { WebCore::ExceptionCode::NotFoundError };
    case FileSystemStorageError::InvalidDataType:
        return WebCore::Exception { WebCore::ExceptionCode::TypeError, "Data type is invalid"_s };
    case FileSystemStorageError::InvalidModification:
        return WebCore::Exception { WebCore::ExceptionCode::InvalidModificationError };
    case FileSystemStorageError::InvalidName:
        return WebCore::Exception { WebCore::ExceptionCode::TypeError, "Name is invalid"_s };
    case FileSystemStorageError::InvalidState:
        return WebCore::Exception { WebCore::ExceptionCode::InvalidStateError };
    case FileSystemStorageError::MissingArgument:
        return WebCore::Exception { WebCore::ExceptionCode::TypeError, "Required argument is missing"_s };
    case FileSystemStorageError::TypeMismatch:
        return WebCore::Exception { WebCore::ExceptionCode::TypeMismatchError, "File type is incompatible with handle type"_s };
    case FileSystemStorageError::Unknown:
        break;
    }

    return WebCore::Exception { WebCore::ExceptionCode::UnknownError };
}

inline WebCore::ExceptionOr<void> convertToExceptionOr(std::optional<FileSystemStorageError> error)
{
    if (!error)
        return { };

    return convertToException(*error);
}

} // namespace WebKit
