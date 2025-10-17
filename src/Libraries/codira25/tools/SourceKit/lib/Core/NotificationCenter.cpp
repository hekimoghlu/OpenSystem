/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 16, 2022.
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

//===--- NotificationCenter.cpp -------------------------------------------===//
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

#include "SourceKit/Core/NotificationCenter.h"
#include "SourceKit/Core/LangSupport.h"
#include "SourceKit/Support/Concurrency.h"

using namespace SourceKit;

NotificationCenter::NotificationCenter(bool dispatchToMain)
  : DispatchToMain(dispatchToMain) {
}
NotificationCenter::~NotificationCenter() {}

void NotificationCenter::addDocumentUpdateNotificationReceiver(
    DocumentUpdateNotificationReceiver Receiver) {

  toolchain::sys::ScopedLock L(Mtx);
  DocUpdReceivers.push_back(Receiver);
}

void NotificationCenter::addTestNotificationReceiver(
    std::function<void(void)> Receiver) {
  toolchain::sys::ScopedLock L(Mtx);
  TestReceivers.push_back(std::move(Receiver));
}
void NotificationCenter::addSemaEnabledNotificationReceiver(
    std::function<void(void)> Receiver) {
  toolchain::sys::ScopedLock L(Mtx);
  SemaEnabledReceivers.push_back(std::move(Receiver));
}
void NotificationCenter::addCompileWillStartNotificationReceiver(
    CompileWillStartNotificationReceiver Receiver) {
  toolchain::sys::ScopedLock L(Mtx);
  CompileWillStartReceivers.push_back(std::move(Receiver));
}
void NotificationCenter::addCompileDidFinishNotificationReceiver(
    CompileDidFinishNotificationReceiver Receiver) {
  toolchain::sys::ScopedLock L(Mtx);
  CompileDidFinishReceivers.push_back(std::move(Receiver));
}

#define POST_NOTIFICATION(Receivers, Args...)                                  \
  do {                                                                         \
    decltype(Receivers) recvs;                                                 \
    {                                                                          \
      toolchain::sys::ScopedLock L(Mtx);                                            \
      recvs = Receivers;                                                       \
    }                                                                          \
    auto sendNote = [=] {                                                      \
      for (auto &Fn : recvs)                                                   \
        Fn(Args);                                                              \
    };                                                                         \
    if (DispatchToMain)                                                        \
      WorkQueue::dispatchOnMain(sendNote);                                     \
    else                                                                       \
      sendNote();                                                              \
  } while (0)

void NotificationCenter::postDocumentUpdateNotification(
    StringRef DocumentName) const {
  std::string docName = DocumentName.str();
  POST_NOTIFICATION(DocUpdReceivers, docName);
}
void NotificationCenter::postTestNotification() const {
  POST_NOTIFICATION(TestReceivers, );
}
void NotificationCenter::postSemaEnabledNotification() const {
  POST_NOTIFICATION(SemaEnabledReceivers, );
}
void NotificationCenter::postCompileWillStartNotification(
    uint64_t CompileID, trace::OperationKind OpKind,
    const trace::CodiraInvocation &Inv) const {
  trace::CodiraInvocation inv(Inv);
  POST_NOTIFICATION(CompileWillStartReceivers, CompileID, OpKind, inv);
}
void NotificationCenter::postCompileDidFinishNotification(
    uint64_t CompileID, trace::OperationKind OpKind,
    ArrayRef<DiagnosticEntryInfo> Diagnostics) const {
  std::vector<DiagnosticEntryInfo> diags(Diagnostics);
  POST_NOTIFICATION(CompileDidFinishReceivers, CompileID, OpKind, diags);
}
