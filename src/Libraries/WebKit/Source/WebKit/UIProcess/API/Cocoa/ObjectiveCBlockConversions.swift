/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 23, 2023.
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
/// A family of conversions for translating between Swift blocks expecting a `Result<V, Error>` and
/// Objective-C callbacks of the form `(T?, Error?)`.
///
/// Depending on the semantics of the Objective-C API, only one of these conversions is appropriate.
/// - If the Objective-C block expects to be called with exactly one non-null argument, use the
/// `exclusive(_:)` conversion.
/// - If the Objective-C block can be called with one or zero non-null arguments, use the
/// `treatNilAsSuccess` conversion.
/// - If the Objective-C block can be called with two non-null values, it is ineligible for
/// conversion to a `Result`.
/// - `boxingNilAsAnyForCompatibility` exists as a workaround for http://webkit.org/b/216198, and
/// should not be used by new code.
///
/// Following WebKit API conventions, closures returned are isolated to MainActor so that they can
/// be safely called from the main thread.
enum ObjCBlockConversion {
    /// Converts a block from `(Result<Value, Error>) -> Void` to `(Value?, Error?) -> Void`.
    ///
    /// The result block must be called with exactly one non-null argument. If both arguments are
    /// non-null then `handler` will be called with `.success(T)`. If both arguments are `nil`
    /// the conversion will trap.
    static nonisolated func exclusive<Value>(_ handler: @MainActor @escaping (Result<Value, Error>) -> Void) -> @MainActor (Value?, Error?) -> Void {
        return { value, error in
            if let value = value {
                handler(.success(value))
            } else if let error = error {
                handler(.failure(error))
            } else {
                preconditionFailure("Bug in WebKit: Received neither result or failure.")
            }
        }
    }

    /// Converts a block from `(Result<Value?, Error>) -> Void` to `(Value?, Error) -> Void`.
    ///
    /// This performs the same conversion as `Self.exclusive(_:)`, but if the result block is called
    /// with `(nil, nil)` then `handler` is called with `.success(nil)`.
    static nonisolated func treatNilAsSuccess<Value>(_ handler: @MainActor @escaping (Result<Value?, Error>) -> Void) -> @MainActor (Value?, Error?) -> Void {
        return { value, error in
            if let error = error {
                handler(.failure(error))
            } else {
                handler(.success(value))
            }
        }
    }

    /// Converts a block from `(Result<Value, Error>) -> Void` to `(Value?, Error) -> Void`.
    ///
    /// This performs the same conversion as `Self.exclusive(_:)`, but if the result block is called
    /// with `(nil, nil)` then `handler` is called with `.success(Optional<Any>.none as Any)`. This
    /// is a compatibility behavior for http://webkit.org/b/216198, and should not be adopted by
    /// new code.
    static nonisolated func boxingNilAsAnyForCompatibility(_ handler: @MainActor @escaping (Result<Any, Error>) -> Void) -> @MainActor (Any?, Error?) -> Void {
        return { value, error in
            if let error = error {
                handler(.failure(error))
            } else if let success = value {
                handler(.success(success))
            } else {
                handler(.success(Optional<Any>.none as Any))
            }
        }
    }
}
