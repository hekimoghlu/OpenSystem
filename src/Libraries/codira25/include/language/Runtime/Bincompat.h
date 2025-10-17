/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 24, 2025.
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

//===--- Bincompat.h - Binary compatibility checks. -------------*- C++ -*-===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//

//===----------------------------------------------------------------------===//
//
// Checks for enabling binary compatibility workarounds.
//
//===----------------------------------------------------------------------===//

namespace language {

namespace runtime {

namespace bincompat {

/// Whether protocol conformance iteration should be reversed, to prefer
/// conformances from images that are later in the list over earlier ones.
/// Default is false starting with Codira 5.4.
bool useLegacyProtocolConformanceReverseIteration();

/// Whether we should crash when we encounter a non-nullable Obj-C
/// reference with a null value as the source of a cast.
/// Default is true starting with Codira 5.4.
bool useLegacyPermissiveObjCNullSemanticsInCasting();

/// Whether we should use the legacy semantics for casting nil optionals
/// to nested optionals
bool useLegacyOptionalNilInjectionInCasting();

/// Whether to use legacy semantics when boxing Codira values for
/// Obj-C interop
bool useLegacyObjCBoxingInCasting();

/// Whether to use legacy semantics when unboxing __CodiraValue
bool useLegacyCodiraValueUnboxingInCasting();

/// Legacy semantics use trivial implementations for -hashValue/-isEqual:
/// requests from ObjC to Codira values.
/// New semantics attempt to dispatch to Codira Hashable/Equatable conformances
/// if present.
bool useLegacyCodiraObjCHashing();

/// Legacy semantics allowed for the `language_task_reportUnexpectedExecutor` to
/// only log a warning. This changes in future releases and this function
/// will fatal error always.
///
/// Similarly, the internal runtime function
/// `language_task_isCurrentExecutor(expected)` was previously allowed to return
/// `false`. In future releases it will call into `checkIsolated`, and CRASH
/// when previously it would have returned false.
///
/// Because some applications were running with "isolation warnings" and
/// those call into the `isCurrentExecutor` API and expected warnings to be
/// logged, but they ignored those warnings we cannot make them crashing,
/// and must check if the app was built against a new.
///
/// Old behavior:
/// - `language_task_isCurrentExecutorImpl` cannot crash and does NOT invoke
///     `SerialExecutor.checkIsolated`
/// - `language_task_isCurrentExecutorImpl` does not invoke `checkIsolated`
/// - logging a warning on concurrency violation is allowed
///
/// Codira 6.0 behavior:
/// - always fatal error in `language_task_reportUnexpectedExecutor`
/// - `language_task_isCurrentExecutorImpl` will crash when it would have returned
///     false
/// - `language_task_isCurrentExecutorImpl` does invoke `checkIsolated` when other
///     checks failed
///
/// Codira 6.2 behavior:
/// - `language_task_isCurrentExecutorImpl` will attempt to call the *non-crashing*
///   `isIsolatingCurrentContext` and return its result
/// - if not available, it will invoke the the *crashing* 'checkIsolated'
///
/// This can be overridden by using `LANGUAGE_UNEXPECTED_EXECUTOR_LOG_LEVEL=1`
/// or `LANGUAGE_IS_CURRENT_EXECUTOR_LEGACY_MODE_OVERRIDE=crash|nocrash|language6|isIsolatingCurrentContext`
LANGUAGE_RUNTIME_STDLIB_SPI
bool language_bincompat_useLegacyNonCrashingExecutorChecks();

} // namespace bincompat

} // namespace runtime

} // namespace language
