/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 1, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_AGC2_LIMITER_DB_GAIN_CURVE_H_
#define MODULES_AUDIO_PROCESSING_AGC2_LIMITER_DB_GAIN_CURVE_H_

#include <array>

#include "modules/audio_processing/agc2/agc2_testing_common.h"

namespace webrtc {

// A class for computing a limiter gain curve (in dB scale) given a set of
// hard-coded parameters (namely, kLimiterDbGainCurveMaxInputLevelDbFs,
// kLimiterDbGainCurveKneeSmoothnessDb, and
// kLimiterDbGainCurveCompressionRatio). The generated curve consists of four
// regions: identity (linear), knee (quadratic polynomial), compression
// (linear), saturation (linear). The aforementioned constants are used to shape
// the different regions.
class LimiterDbGainCurve {
 public:
  LimiterDbGainCurve();

  double max_input_level_db() const { return max_input_level_db_; }
  double max_input_level_linear() const { return max_input_level_linear_; }
  double knee_start_linear() const { return knee_start_linear_; }
  double limiter_start_linear() const { return limiter_start_linear_; }

  // These methods can be marked 'constexpr' in C++ 14.
  double GetOutputLevelDbfs(double input_level_dbfs) const;
  double GetGainLinear(double input_level_linear) const;
  double GetGainFirstDerivativeLinear(double x) const;
  double GetGainIntegralLinear(double x0, double x1) const;

 private:
  double GetKneeRegionOutputLevelDbfs(double input_level_dbfs) const;
  double GetCompressorRegionOutputLevelDbfs(double input_level_dbfs) const;

  static constexpr double max_input_level_db_ = test::kLimiterMaxInputLevelDbFs;
  static constexpr double knee_smoothness_db_ = test::kLimiterKneeSmoothnessDb;
  static constexpr double compression_ratio_ = test::kLimiterCompressionRatio;

  const double max_input_level_linear_;

  // Do not modify signal with level <= knee_start_dbfs_.
  const double knee_start_dbfs_;
  const double knee_start_linear_;

  // The upper end of the knee region, which is between knee_start_dbfs_ and
  // limiter_start_dbfs_.
  const double limiter_start_dbfs_;
  const double limiter_start_linear_;

  // Coefficients {a, b, c} of the knee region polynomial
  // ax^2 + bx + c in the DB scale.
  const std::array<double, 3> knee_region_polynomial_;

  // Parameters for the computation of the first derivative of GetGainLinear().
  const double gain_curve_limiter_d1_;
  const double gain_curve_limiter_d2_;

  // Parameters for the computation of the integral of GetGainLinear().
  const double gain_curve_limiter_i1_;
  const double gain_curve_limiter_i2_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AGC2_LIMITER_DB_GAIN_CURVE_H_
