/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 16, 2023.
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

namespace WebCore {

enum class ExceptionCode : uint8_t {
    // DOMException error names (https://webidl.spec.whatwg.org/#idl-DOMException-error-names).
    // Those need to be kept in sync with the array in DOMException.cpp.
    IndexSizeError, // Deprecated. Use RangeError instead.
    HierarchyRequestError,
    WrongDocumentError,
    InvalidCharacterError,
    NoModificationAllowedError,
    NotFoundError,
    NotSupportedError,
    InUseAttributeError,
    InvalidStateError,
    SyntaxError,
    InvalidModificationError,
    NamespaceError,
    InvalidAccessError, // Deprecated. use NotAllowedError instead.
    TypeMismatchError, // Deprecated. Use TypeError instead.
    SecurityError,
    NetworkError,
    AbortError,
    URLMismatchError,
    QuotaExceededError,
    TimeoutError,
    InvalidNodeTypeError,
    DataCloneError,
    EncodingError,
    NotReadableError,
    UnknownError,
    ConstraintError,
    DataError,
    TransactionInactiveError,
    ReadonlyError,
    VersionError,
    OperationError,
    NotAllowedError,

    // Simple exceptions (https://webidl.spec.whatwg.org/#idl-exceptions).
    RangeError,
    TypeError,
    JSSyntaxError, // Different from DOM SYNTAX_ERR.

    // Non-standard error.
    StackOverflowError,
    OutOfMemoryError,

    // Used to indicate to the bindings that a JS exception was thrown below and it should be propagated.
    ExistingExceptionError,
};

} // namespace WebCore
