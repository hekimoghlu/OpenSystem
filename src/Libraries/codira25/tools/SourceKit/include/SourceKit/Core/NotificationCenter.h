/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 5, 2025.
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

//===--- NotificationCenter.h - ---------------------------------*- C++ -*-===//
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

#ifndef TOOLCHAIN_SOURCEKIT_CORE_NOTIFICATIONCENTER_H
#define TOOLCHAIN_SOURCEKIT_CORE_NOTIFICATIONCENTER_H

#include "SourceKit/Core/Toolchain.h"
#include "SourceKit/Support/Tracing.h"
#include "SourceKit/Support/UIdent.h"
#include "toolchain/Support/Mutex.h"
#include <functional>
#include <vector>

namespace SourceKit {

struct DiagnosticEntryInfo;

typedef std::function<void(StringRef DocumentName)>
    DocumentUpdateNotificationReceiver;

typedef std::function<void(uint64_t CompileID, trace::OperationKind,
                           const trace::CodiraInvocation &)>
    CompileWillStartNotificationReceiver;
typedef std::function<void(uint64_t CompileID, trace::OperationKind,
                           ArrayRef<DiagnosticEntryInfo>)>
    CompileDidFinishNotificationReceiver;

class NotificationCenter {
  bool DispatchToMain;
  std::vector<DocumentUpdateNotificationReceiver> DocUpdReceivers;
  std::vector<std::function<void(void)>> TestReceivers;
  std::vector<std::function<void(void)>> SemaEnabledReceivers;
  std::vector<CompileWillStartNotificationReceiver> CompileWillStartReceivers;
  std::vector<CompileDidFinishNotificationReceiver> CompileDidFinishReceivers;
  mutable toolchain::sys::Mutex Mtx;

public:
  explicit NotificationCenter(bool dispatchToMain);
  ~NotificationCenter();

  void addDocumentUpdateNotificationReceiver(
      DocumentUpdateNotificationReceiver Receiver);
  void addTestNotificationReceiver(std::function<void(void)> Receiver);
  void addSemaEnabledNotificationReceiver(std::function<void(void)> Receiver);
  void addCompileWillStartNotificationReceiver(
      CompileWillStartNotificationReceiver Receiver);
  void addCompileDidFinishNotificationReceiver(
      CompileDidFinishNotificationReceiver Receiver);

  void postDocumentUpdateNotification(StringRef DocumentName) const;
  void postTestNotification() const;
  void postSemaEnabledNotification() const;
  void
  postCompileWillStartNotification(uint64_t CompileID,
                                   trace::OperationKind OpKind,
                                   const trace::CodiraInvocation &Inv) const;
  void postCompileDidFinishNotification(
      uint64_t CompileID, trace::OperationKind OpKind,
      ArrayRef<DiagnosticEntryInfo> Diagnostics) const;
};

} // namespace SourceKit

#endif
