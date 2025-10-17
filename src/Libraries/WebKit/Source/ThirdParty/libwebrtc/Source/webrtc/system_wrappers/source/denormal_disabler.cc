/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 26, 2024.
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
#include "system_wrappers/include/denormal_disabler.h"

#include "rtc_base/checks.h"

namespace webrtc {
namespace {

#if defined(WEBRTC_ARCH_X86_FAMILY) && defined(__clang__)
#define WEBRTC_DENORMAL_DISABLER_X86_SUPPORTED
#endif

#if defined(WEBRTC_DENORMAL_DISABLER_X86_SUPPORTED) || \
    defined(WEBRTC_ARCH_ARM_FAMILY)
#define WEBRTC_DENORMAL_DISABLER_SUPPORTED
#endif

constexpr int kUnspecifiedStatusWord = -1;

#if defined(WEBRTC_DENORMAL_DISABLER_SUPPORTED)

// Control register bit mask to disable denormals on the hardware.
#if defined(WEBRTC_DENORMAL_DISABLER_X86_SUPPORTED)
// On x86 two bits are used: flush-to-zero (FTZ) and denormals-are-zero (DAZ).
constexpr int kDenormalBitMask = 0x8040;
#elif defined(WEBRTC_ARCH_ARM_FAMILY)
// On ARM one bit is used: flush-to-zero (FTZ).
constexpr int kDenormalBitMask = 1 << 24;
#endif

// Reads the relevant CPU control register and returns its value for supported
// architectures and compilers. Otherwise returns `kUnspecifiedStatusWord`.
int ReadStatusWord() {
  int result = kUnspecifiedStatusWord;
#if defined(WEBRTC_DENORMAL_DISABLER_X86_SUPPORTED)
  asm volatile("stmxcsr %0" : "=m"(result));
#elif defined(WEBRTC_ARCH_ARM_FAMILY) && defined(WEBRTC_ARCH_32_BITS)
  asm volatile("vmrs %[result], FPSCR" : [result] "=r"(result));
#elif defined(WEBRTC_ARCH_ARM_FAMILY) && defined(WEBRTC_ARCH_64_BITS)
  asm volatile("mrs %x[result], FPCR" : [result] "=r"(result));
#endif
  return result;
}

// Writes `status_word` in the relevant CPU control register if the architecture
// and the compiler are supported.
void SetStatusWord(int status_word) {
#if defined(WEBRTC_DENORMAL_DISABLER_X86_SUPPORTED)
  asm volatile("ldmxcsr %0" : : "m"(status_word));
#elif defined(WEBRTC_ARCH_ARM_FAMILY) && defined(WEBRTC_ARCH_32_BITS)
  asm volatile("vmsr FPSCR, %[src]" : : [src] "r"(status_word));
#elif defined(WEBRTC_ARCH_ARM_FAMILY) && defined(WEBRTC_ARCH_64_BITS)
  asm volatile("msr FPCR, %x[src]" : : [src] "r"(status_word));
#endif
}

// Returns true if the status word indicates that denormals are enabled.
constexpr bool DenormalsEnabled(int status_word) {
  return (status_word & kDenormalBitMask) != kDenormalBitMask;
}

#endif  // defined(WEBRTC_DENORMAL_DISABLER_SUPPORTED)

}  // namespace

#if defined(WEBRTC_DENORMAL_DISABLER_SUPPORTED)
DenormalDisabler::DenormalDisabler() : DenormalDisabler(/*enabled=*/true) {}

DenormalDisabler::DenormalDisabler(bool enabled)
    : status_word_(enabled ? ReadStatusWord() : kUnspecifiedStatusWord),
      disabling_activated_(enabled && DenormalsEnabled(status_word_)) {
  if (disabling_activated_) {
    RTC_DCHECK_NE(status_word_, kUnspecifiedStatusWord);
    SetStatusWord(status_word_ | kDenormalBitMask);
    RTC_DCHECK(!DenormalsEnabled(ReadStatusWord()));
  }
}

bool DenormalDisabler::IsSupported() {
  return true;
}

DenormalDisabler::~DenormalDisabler() {
  if (disabling_activated_) {
    RTC_DCHECK_NE(status_word_, kUnspecifiedStatusWord);
    SetStatusWord(status_word_);
  }
}
#else
DenormalDisabler::DenormalDisabler() : DenormalDisabler(/*enabled=*/false) {}

DenormalDisabler::DenormalDisabler(bool enabled)
    : status_word_(kUnspecifiedStatusWord), disabling_activated_(false) {}

bool DenormalDisabler::IsSupported() {
  return false;
}

DenormalDisabler::~DenormalDisabler() = default;
#endif

}  // namespace webrtc
