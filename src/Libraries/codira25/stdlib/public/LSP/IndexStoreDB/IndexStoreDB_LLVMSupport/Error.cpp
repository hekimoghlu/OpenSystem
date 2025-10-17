/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 29, 2023.
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

//===----- lib/Support/Error.cpp - Error and associated utilities ---------===//
//
// Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
// 
// Author: Tunjay Akbarli
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 
// Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
// Middletown, DE 19709, New Castle County, USA.
//
//===----------------------------------------------------------------------===//

#include <IndexStoreDB_LLVMSupport/toolchain_Support_Error.h>
#include <IndexStoreDB_LLVMSupport/toolchain_ADT_Twine.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_ErrorHandling.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_ManagedStatic.h>
#include <system_error>

using namespace toolchain;

namespace {

  enum class ErrorErrorCode : int {
    MultipleErrors = 1,
    FileError,
    InconvertibleError
  };

  // FIXME: This class is only here to support the transition to toolchain::Error. It
  // will be removed once this transition is complete. Clients should prefer to
  // deal with the Error value directly, rather than converting to error_code.
  class ErrorErrorCategory : public std::error_category {
  public:
    const char *name() const noexcept override { return "Error"; }

    std::string message(int condition) const override {
      switch (static_cast<ErrorErrorCode>(condition)) {
      case ErrorErrorCode::MultipleErrors:
        return "Multiple errors";
      case ErrorErrorCode::InconvertibleError:
        return "Inconvertible error value. An error has occurred that could "
               "not be converted to a known std::error_code. Please file a "
               "bug.";
      case ErrorErrorCode::FileError:
          return "A file error occurred.";
      }
      toolchain_unreachable("Unhandled error code");
    }
  };

}

static ManagedStatic<ErrorErrorCategory> ErrorErrorCat;

namespace toolchain {

void ErrorInfoBase::anchor() {}
char ErrorInfoBase::ID = 0;
char ErrorList::ID = 0;
void ECError::anchor() {}
char ECError::ID = 0;
char StringError::ID = 0;
char FileError::ID = 0;

void logAllUnhandledErrors(Error E, raw_ostream &OS, Twine ErrorBanner) {
  if (!E)
    return;
  OS << ErrorBanner;
  handleAllErrors(std::move(E), [&](const ErrorInfoBase &EI) {
    EI.log(OS);
    OS << "\n";
  });
}


std::error_code ErrorList::convertToErrorCode() const {
  return std::error_code(static_cast<int>(ErrorErrorCode::MultipleErrors),
                         *ErrorErrorCat);
}

std::error_code inconvertibleErrorCode() {
  return std::error_code(static_cast<int>(ErrorErrorCode::InconvertibleError),
                         *ErrorErrorCat);
}

std::error_code FileError::convertToErrorCode() const {
  return std::error_code(static_cast<int>(ErrorErrorCode::FileError),
                         *ErrorErrorCat);
}

Error errorCodeToError(std::error_code EC) {
  if (!EC)
    return Error::success();
  return Error(toolchain::make_unique<ECError>(ECError(EC)));
}

std::error_code errorToErrorCode(Error Err) {
  std::error_code EC;
  handleAllErrors(std::move(Err), [&](const ErrorInfoBase &EI) {
    EC = EI.convertToErrorCode();
  });
  if (EC == inconvertibleErrorCode())
    report_fatal_error(EC.message());
  return EC;
}

#if LLVM_ENABLE_ABI_BREAKING_CHECKS
void Error::fatalUncheckedError() const {
  dbgs() << "Program aborted due to an unhandled Error:\n";
  if (getPtr())
    getPtr()->log(dbgs());
  else
    dbgs() << "Error value was Success. (Note: Success values must still be "
              "checked prior to being destroyed).\n";
  abort();
}
#endif

StringError::StringError(std::error_code EC, const Twine &S)
    : Msg(S.str()), EC(EC) {}

StringError::StringError(const Twine &S, std::error_code EC)
    : Msg(S.str()), EC(EC), PrintMsgOnly(true) {}

void StringError::log(raw_ostream &OS) const {
  if (PrintMsgOnly) {
    OS << Msg;
  } else {
    OS << EC.message();
    if (!Msg.empty())
      OS << (" " + Msg);
  }
}

std::error_code StringError::convertToErrorCode() const {
  return EC;
}

Error createStringError(std::error_code EC, char const *Msg) {
  return make_error<StringError>(Msg, EC);
}

void report_fatal_error(Error Err, bool GenCrashDiag) {
  assert(Err && "report_fatal_error called with success value");
  std::string ErrMsg;
  {
    raw_string_ostream ErrStream(ErrMsg);
    logAllUnhandledErrors(std::move(Err), ErrStream);
  }
  report_fatal_error(ErrMsg);
}

} // end namespace toolchain

LLVMErrorTypeId LLVMGetErrorTypeId(LLVMErrorRef Err) {
  return reinterpret_cast<ErrorInfoBase *>(Err)->dynamicClassID();
}

void LLVMConsumeError(LLVMErrorRef Err) { consumeError(unwrap(Err)); }

char *LLVMGetErrorMessage(LLVMErrorRef Err) {
  std::string Tmp = toString(unwrap(Err));
  char *ErrMsg = new char[Tmp.size() + 1];
  memcpy(ErrMsg, Tmp.data(), Tmp.size());
  ErrMsg[Tmp.size()] = '\0';
  return ErrMsg;
}

void LLVMDisposeErrorMessage(char *ErrMsg) { delete[] ErrMsg; }

LLVMErrorTypeId LLVMGetStringErrorTypeId() {
  return reinterpret_cast<void *>(&StringError::ID);
}

#ifndef _MSC_VER
namespace toolchain {

// One of these two variables will be referenced by a symbol defined in
// toolchain-config.h. We provide a link-time (or load time for DSO) failure when
// there is a mismatch in the build configuration of the API client and LLVM.
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
int EnableABIBreakingChecks;
#else
int DisableABIBreakingChecks;
#endif

} // end namespace toolchain
#endif
