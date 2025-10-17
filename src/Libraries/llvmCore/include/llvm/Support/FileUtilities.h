/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 7, 2024.
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

//===- llvm/Support/FileUtilities.h - File System Utilities -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a family of utility functions which are useful for doing
// various things with files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_FILEUTILITIES_H
#define LLVM_SUPPORT_FILEUTILITIES_H

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace llvm {

  /// DiffFilesWithTolerance - Compare the two files specified, returning 0 if
  /// the files match, 1 if they are different, and 2 if there is a file error.
  /// This function allows you to specify an absolute and relative FP error that
  /// is allowed to exist.  If you specify a string to fill in for the error
  /// option, it will set the string to an error message if an error occurs, or
  /// if the files are different.
  ///
  int DiffFilesWithTolerance(const sys::PathWithStatus &FileA,
                             const sys::PathWithStatus &FileB,
                             double AbsTol, double RelTol,
                             std::string *Error = 0);


  /// FileRemover - This class is a simple object meant to be stack allocated.
  /// If an exception is thrown from a region, the object removes the filename
  /// specified (if deleteIt is true).
  ///
  class FileRemover {
    SmallString<128> Filename;
    bool DeleteIt;
  public:
    FileRemover() : DeleteIt(false) {}

    explicit FileRemover(const Twine& filename, bool deleteIt = true)
      : DeleteIt(deleteIt) {
      filename.toVector(Filename);
    }

    ~FileRemover() {
      if (DeleteIt) {
        // Ignore problems deleting the file.
        bool existed;
        sys::fs::remove(Filename.str(), existed);
      }
    }

    /// setFile - Give ownership of the file to the FileRemover so it will
    /// be removed when the object is destroyed.  If the FileRemover already
    /// had ownership of a file, remove it first.
    void setFile(const Twine& filename, bool deleteIt = true) {
      if (DeleteIt) {
        // Ignore problems deleting the file.
        bool existed;
        sys::fs::remove(Filename.str(), existed);
      }

      Filename.clear();
      filename.toVector(Filename);
      DeleteIt = deleteIt;
    }

    /// releaseFile - Take ownership of the file away from the FileRemover so it
    /// will not be removed when the object is destroyed.
    void releaseFile() { DeleteIt = false; }
  };
} // End llvm namespace

#endif
