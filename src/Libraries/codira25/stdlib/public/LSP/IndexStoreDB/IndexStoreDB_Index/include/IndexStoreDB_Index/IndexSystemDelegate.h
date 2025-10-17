/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 6, 2022.
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

//===--- IndexSystemDelegate.h ----------------------------------*- C++ -*-===//
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

#ifndef INDEXSTOREDB_INDEX_INDEXSYSTEMDELEGATE_H
#define INDEXSTOREDB_INDEX_INDEXSYSTEMDELEGATE_H

#include <IndexStoreDB_Index/StoreUnitInfo.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_Chrono.h>
#include <memory>
#include <string>

namespace IndexStoreDB {
namespace index {
struct StoreUnitInfo;
class OutOfDateFileTrigger;

typedef std::shared_ptr<OutOfDateFileTrigger> OutOfDateFileTriggerRef;

/// Records a known out-of-date file path for a unit, along with its
/// modification time. This is used to provide IndexDelegate with information
/// about the file that triggered the unit to become out-of-date.
class OutOfDateFileTrigger final {
  std::string FilePath;
  toolchain::sys::TimePoint<> ModTime;

public:
  explicit OutOfDateFileTrigger(StringRef filePath,
                                toolchain::sys::TimePoint<> modTime)
      : FilePath(filePath), ModTime(modTime) {}

  static OutOfDateFileTriggerRef create(StringRef filePath,
                                        toolchain::sys::TimePoint<> modTime) {
    return std::make_shared<OutOfDateFileTrigger>(filePath, modTime);
  }

  toolchain::sys::TimePoint<> getModTime() const { return ModTime; }

  /// Returns a reference to the stored file path. Note this has the same
  /// lifetime as the trigger.
  StringRef getPathRef() const { return FilePath; }

  std::string getPath() const { return FilePath; }
  std::string description() { return FilePath; }
};

class INDEXSTOREDB_EXPORT IndexSystemDelegate {
public:
  virtual ~IndexSystemDelegate() {}

  /// Called when the datastore gets initialized and receives the number of available units.
  virtual void initialPendingUnits(unsigned numUnits) {}

  virtual void processingAddedPending(unsigned NumActions) {}
  virtual void processingCompleted(unsigned NumActions) {}

  virtual void processedStoreUnit(StoreUnitInfo unitInfo) {}

  virtual void unitIsOutOfDate(StoreUnitInfo unitInfo,
                               OutOfDateFileTriggerRef trigger,
                               bool synchronous = false) {}

private:
  virtual void anchor();
};

} // namespace index
} // namespace IndexStoreDB

#endif
