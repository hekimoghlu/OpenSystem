/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 19, 2025.
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
#ifndef MODULES_AUDIO_PROCESSING_UTILITY_OOURA_FFT_H_
#define MODULES_AUDIO_PROCESSING_UTILITY_OOURA_FFT_H_

#include "rtc_base/system/arch.h"

namespace webrtc {

#if defined(WEBRTC_ARCH_X86_FAMILY)
void cft1st_128_SSE2(float* a);
void cftmdl_128_SSE2(float* a);
void rftfsub_128_SSE2(float* a);
void rftbsub_128_SSE2(float* a);
#endif

#if defined(MIPS_FPU_LE)
void cft1st_128_mips(float* a);
void cftmdl_128_mips(float* a);
void rftfsub_128_mips(float* a);
void rftbsub_128_mips(float* a);
#endif

#if defined(WEBRTC_HAS_NEON)
void cft1st_128_neon(float* a);
void cftmdl_128_neon(float* a);
void rftfsub_128_neon(float* a);
void rftbsub_128_neon(float* a);
#endif

class OouraFft {
 public:
  // Ctor allowing the availability of SSE2 support to be specified.
  explicit OouraFft(bool sse2_available);

  // Deprecated: This Ctor will soon be removed.
  OouraFft();
  ~OouraFft();
  void Fft(float* a) const;
  void InverseFft(float* a) const;

 private:
  void cft1st_128(float* a) const;
  void cftmdl_128(float* a) const;
  void rftfsub_128(float* a) const;
  void rftbsub_128(float* a) const;

  void cftfsub_128(float* a) const;
  void cftbsub_128(float* a) const;
  void bitrv2_128(float* a) const;
  bool use_sse2_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_UTILITY_OOURA_FFT_H_
