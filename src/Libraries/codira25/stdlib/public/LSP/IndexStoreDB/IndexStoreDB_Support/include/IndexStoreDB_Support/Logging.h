/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 10, 2022.
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

//===--- Logging.h - Logging Interface --------------------------*- C++ -*-===//
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

#ifndef LLVM_INDEXSTOREDB_SUPPORT_LOGGING_H
#define LLVM_INDEXSTOREDB_SUPPORT_LOGGING_H

#include <IndexStoreDB_Support/LLVM.h>
#include <IndexStoreDB_Support/Visibility.h>
#include <IndexStoreDB_LLVMSupport/toolchain_ADT_IntrusiveRefCntPtr.h>
#include <IndexStoreDB_LLVMSupport/toolchain_ADT_SmallString.h>
#include <IndexStoreDB_LLVMSupport/toolchain_ADT_StringRef.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_raw_ostream.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_Timer.h>
#include <string>

namespace toolchain {
class format_object_base;
}

namespace IndexStoreDB {
  class Logger;

INDEXSTOREDB_EXPORT void writeEscaped(StringRef Str, raw_ostream &OS);

typedef IntrusiveRefCntPtr<Logger> LogRef;

/// \brief Collects logging output and writes it to stderr when it's destructed.
/// Common use case:
/// \code
///   if (LogRef Log = Logger::make(__func__, Logger::Level::Warning)) {
///     *Log << "stuff";
///   }
/// \endcode
class INDEXSTOREDB_EXPORT Logger : public toolchain::ThreadSafeRefCountedBase<Logger> {
public:
  enum class Level : unsigned char {
    /// \brief No logging.
    None = 0,
    /// \brief Warning level.
    Warning = 1,
    /// \brief Information level for high priority messages.
    InfoHighPrio = 2,
    /// \brief Information level for medium priority messages.
    InfoMediumPrio = 3,
    /// \brief Information level for low priority messages.
    InfoLowPrio = 4
  };

private:
  std::string Name;
  Level CurrLevel;
  SmallString<64> Msg;
  toolchain::raw_svector_ostream LogOS;
  uint64_t thread_id;
  toolchain::TimeRecord TimeR;

  static std::string LoggerName;
  static std::atomic<Level> LoggingLevel;

public:
  static void enableLoggingByEnvVar(const char *EnvVarName,
                                    StringRef LoggerName);

  static bool isLoggingEnabledForLevel(Level LogLevel) {
    return LoggingLevel >= LogLevel;
  }
  static void enableLogging(StringRef Name, Level LogLevel) {
    LoggerName = Name;
    LoggingLevel = LogLevel;
  }

  static void setLogLevelByNum(unsigned LevelNum) {
    LoggingLevel = getLogLevelByNum(LevelNum);
  }
  static Level getLogLevelByNum(unsigned LevelNum);
  static unsigned getCurrentLogLevelNum();

  static LogRef make(toolchain::StringRef Name, Level LogLevel) {
    if (isLoggingEnabledForLevel(LogLevel)) return new Logger(Name, LogLevel);
    return nullptr;
  }

  Logger(StringRef Name, Level LogLevel);
  ~Logger();

  toolchain::raw_ostream &getOS() { return LogOS; }

  Logger &operator<<(toolchain::StringRef Str) { LogOS << Str; return *this; }
  Logger &operator<<(const char *Str) { if (Str) LogOS << Str; return *this; }
  Logger &operator<<(unsigned long N) { LogOS << N; return *this; }
  Logger &operator<<(unsigned long long N) { LogOS << N; return *this; }
  Logger &operator<<(long N) { LogOS << N ; return *this; }
  Logger &operator<<(unsigned int N) { LogOS << N; return *this; }
  Logger &operator<<(int N) { LogOS << N; return *this; }
  Logger &operator<<(char C) { LogOS << C; return *this; }
  Logger &operator<<(unsigned char C) { LogOS << C; return *this; }
  Logger &operator<<(signed char C) { LogOS << C; return *this; }
  Logger &operator<<(const toolchain::format_object_base &Fmt);
};

} // namespace IndexStoreDB.

/// \brief Macros to automate common uses of Logger. Like this:
/// \code
///   LOG_FUNC_SECTION_WARN {
///     *Log << "blah";
///   }
/// \endcode
#define LOG_SECTION(NAME, LEVEL) \
  if (IndexStoreDB::LogRef Log = IndexStoreDB::Logger::make(NAME, IndexStoreDB::Logger::Level::LEVEL))
#define LOG_FUNC_SECTION(LEVEL) LOG_SECTION(__func__, LEVEL)
#define LOG_FUNC_SECTION_WARN LOG_FUNC_SECTION(Warning)

#define LOG(NAME, LEVEL, msg) LOG_SECTION(NAME, LEVEL) \
  do { *Log << msg; } while(0)
#define LOG_FUNC(LEVEL, msg) LOG_FUNC_SECTION(LEVEL) \
  do { *Log << msg; } while(0)
#define LOG_WARN(NAME, msg) LOG(NAME, Warning, msg)
#define LOG_WARN_FUNC(msg) LOG_FUNC(Warning, msg)
#define LOG_INFO_FUNC(PRIO, msg) LOG_FUNC(Info##PRIO##Prio, msg)
#define LOG_INFO(NAME, PRIO, msg) LOG(NAME, Info##PRIO##Prio, msg)

#endif
