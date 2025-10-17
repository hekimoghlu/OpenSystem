/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 9, 2022.
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

//===--- UIDHandling.cpp --------------------------------------------------===//
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

#include "SourceKit/Support/UIdent.h"
#include "sourcekitd/Internal.h"

#include "toolchain/Support/Mutex.h"

#include <Block.h>

using namespace SourceKit;

static toolchain::sys::Mutex GlobalHandlersMtx;
static sourcekitd_uid_handler_t UidMappingHandler;
static sourcekitd_str_from_uid_handler_t StrFromUidMappingHandler;

void sourcekitd_set_uid_handler(sourcekitd_uid_handler_t handler) {
  toolchain::sys::ScopedLock L(GlobalHandlersMtx);
  sourcekitd_uid_handler_t newHandler = Block_copy(handler);
  Block_release(UidMappingHandler);
  UidMappingHandler = newHandler;
}

void sourcekitd_set_uid_handlers(
    sourcekitd_uid_from_str_handler_t uid_from_str,
    sourcekitd_str_from_uid_handler_t str_from_uid) {
  toolchain::sys::ScopedLock L(GlobalHandlersMtx);

  sourcekitd_uid_handler_t newUIDFromStrHandler = Block_copy(uid_from_str);
  Block_release(UidMappingHandler);
  UidMappingHandler = newUIDFromStrHandler;

  sourcekitd_str_from_uid_handler_t newStrFromUIDHandler =
      Block_copy(str_from_uid);
  Block_release(StrFromUidMappingHandler);
  StrFromUidMappingHandler = newStrFromUIDHandler;
}

sourcekitd_uid_t sourcekitd::SKDUIDFromUIdent(UIdent UID) {
  if (void *Tag = UID.getTag())
    return reinterpret_cast<sourcekitd_uid_t>(Tag);

  if (UidMappingHandler) {
    sourcekitd_uid_t skduid = UidMappingHandler(UID.c_str());
    if (skduid) {
      UID.setTag(skduid);
      return skduid;
    }
  }

  return reinterpret_cast<sourcekitd_uid_t>(UID.getAsOpaqueValue());
}

UIdent sourcekitd::UIdentFromSKDUID(sourcekitd_uid_t uid) {
  if (StrFromUidMappingHandler)
    return UIdent(StrFromUidMappingHandler(uid));

  return UIdent::getFromOpaqueValue(uid);
}
