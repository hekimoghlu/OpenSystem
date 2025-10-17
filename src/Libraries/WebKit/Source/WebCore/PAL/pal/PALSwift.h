/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 4, 2023.
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

#include <cstdint>
#include <wtf/Vector.h>

namespace Cpp {

using VectorUInt8 = WTF::Vector<uint8_t>;
using SpanConstUInt8 = std::span<const uint8_t>;
using OptionalVectorUInt8 = std::optional<WTF::Vector<uint8_t>>;

enum class ErrorCodes: int {
    Success = 0,
    WrongTagSize,
    EncryptionFailed,
    EncryptionResultNil,
    InvalidArgument,
    TooBigArguments,
    DecryptionFailed,
    HashingFailed,
    PublicKeyProvidedToSign,
    FailedToSign,
    FailedToVerify,
    PrivateKeyProvidedForVerification,
    FailedToImport,
    FailedToDerive,
    FailedToExport,
    DefaultValue,
    UnsupportedAlgorithm,
};
struct CryptoOperationReturnValue {
    ErrorCodes errorCode = ErrorCodes::DefaultValue;
    VectorUInt8 result;
};

} // Cpp

#ifndef __swift__
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include "PALSwift-Generated.h"
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END
#endif
