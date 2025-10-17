/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 16, 2023.
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

//===--- ExecutorBridge.h - C++ side of executor bridge -------------------===//
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

#ifndef LANGUAGE_EXECUTOR_BRIDGE_H_
#define LANGUAGE_EXECUTOR_BRIDGE_H_

#include "language/Runtime/Concurrency.h"

namespace language {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"

extern "C" LANGUAGE_CC(language)
SerialExecutorRef language_getMainExecutor();

#if !LANGUAGE_CONCURRENCY_EMBEDDED
extern "C" LANGUAGE_CC(language)
void *language_createDispatchEventC(void (*handler)(void *), void *context);

extern "C" LANGUAGE_CC(language)
void language_destroyDispatchEventC(void *event);

extern "C" LANGUAGE_CC(language)
void language_signalDispatchEvent(void *event);
#endif // !LANGUAGE_CONCURRENCY_EMBEDDED

extern "C" LANGUAGE_CC(language) __attribute__((noreturn))
void language_dispatchMain();

extern "C" LANGUAGE_CC(language)
void language_createDefaultExecutors();

extern "C" LANGUAGE_CC(language)
void language_createDefaultExecutorsOnce();

#pragma clang diagnostic pop

} // namespace language

#endif /* LANGUAGE_EXECUTOR_BRIDGE_H_ */
