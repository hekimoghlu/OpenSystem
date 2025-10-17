/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 19, 2024.
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

//===- llvm/Support/ThreadLocal.h - Thread Local Data ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the llvm::sys::ThreadLocal class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_THREAD_LOCAL_H
#define LLVM_SYSTEM_THREAD_LOCAL_H

#include "llvm/Support/Threading.h"
#include "llvm/Support/DataTypes.h"
#include <cassert>

namespace llvm {
  namespace sys {
    // ThreadLocalImpl - Common base class of all ThreadLocal instantiations.
    // YOU SHOULD NEVER USE THIS DIRECTLY.
    class ThreadLocalImpl {
      typedef uint64_t ThreadLocalDataTy;
      /// \brief Platform-specific thread local data.
      ///
      /// This is embedded in the class and we avoid malloc'ing/free'ing it,
      /// to make this class more safe for use along with CrashRecoveryContext.
      union {
        char data[sizeof(ThreadLocalDataTy)];
        ThreadLocalDataTy align_data;
      };
    public:
      ThreadLocalImpl();
      virtual ~ThreadLocalImpl();
      void setInstance(const void* d);
      const void* getInstance();
      void removeInstance();
    };

    /// ThreadLocal - A class used to abstract thread-local storage.  It holds,
    /// for each thread, a pointer a single object of type T.
    template<class T>
    class ThreadLocal : public ThreadLocalImpl {
    public:
      ThreadLocal() : ThreadLocalImpl() { }

      /// get - Fetches a pointer to the object associated with the current
      /// thread.  If no object has yet been associated, it returns NULL;
      T* get() { return static_cast<T*>(getInstance()); }

      // set - Associates a pointer to an object with the current thread.
      void set(T* d) { setInstance(d); }

      // erase - Removes the pointer associated with the current thread.
      void erase() { removeInstance(); }
    };
  }
}

#endif
