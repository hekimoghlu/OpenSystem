/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 20, 2024.
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

//===--- Internal-XPC.h - ---------------------------------------*- C++ -*-===//
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

#ifndef TOOLCHAIN_SOURCEKITD_INTERNAL_XPC_H
#define TOOLCHAIN_SOURCEKITD_INTERNAL_XPC_H

#include "sourcekitd/Internal.h"

namespace sourcekitd {
namespace xpc {

static const char *KeyMsg = "msg";
static const char *KeyInternalMsg = "internal_msg";
static const char *KeySemaEditorDelay = "semantic_editor_delay";
static const char *KeyTracingEnabled = "tracing_enabled";
static const char *KeyMsgResponse = "response";
static const char *KeyCancelToken = "cancel_token";
static const char *KeyCancelRequest = "cancel_request";
static const char *KeyDisposeRequestHandle = "dispose_request_handle";
static const char *KeyPlugins = "plugins";

enum class Message {
  Initialization,
  Notification,
  UIDSynchronization,
};

} // namespace xpc
} // namespace sourcekitd

#endif
